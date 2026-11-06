"""Task-aware training schedules. Numerical optimization quality is out of scope."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from math import ceil

from .core import Allocation, DTYPE_BYTES, Event, Graph


def rl_lengths(rl):
    prompts = rl['prompt_lengths'] or [rl['prompt_length']] * rl['prompts_per_step']
    prompts = [length for length in prompts for _ in range(rl['samples_per_prompt'])]
    responses = rl['response_lengths'] or [rl['response_length']] * len(prompts)
    return prompts, responses


def validate_training_task(config):
    """Reject unsupported combinations during validate, before constructing a DAG."""
    w, e, task = config.workload, config.execution, config.workload.training
    if w.scenario != 'training' or config.model.architecture is None:
        raise ValueError('workload.training requires scenario=training and model.architecture')
    if config.model.architecture.output_head_dim not in (None, config.model.architecture.vocab_size):
        raise ValueError('The training actor requires a vocabulary output head')
    if task.kind in ('pretrain', 'sft'):
        lengths = task.data.sequence_lengths or [w.sequence_length] * w.batch_size
        if len(lengths) != w.batch_size:
            raise ValueError('training.data.sequence_lengths must contain batch_size lengths')
        if task.kind == 'sft':
            labels = task.sft.supervised_tokens
            if task.sft.loss_mask == 'response':
                if len(labels) != len(lengths) or any(n < 0 or n > s for n, s in zip(labels, lengths)) or not sum(labels):
                    raise ValueError('SFT response masking requires one supervised_tokens count per sequence, within valid lengths and a positive total')
            elif labels:
                raise ValueError('SFT loss_mask=all must not supply supervised_tokens')
    elif task.kind == 'dpo':
        if any(len(values) != w.batch_size for values in (task.dpo.chosen_lengths, task.dpo.rejected_lengths)):
            raise ValueError('DPO chosen_lengths/rejected_lengths each require batch_size entries (pairs per microbatch)')
        lengths = task.dpo.chosen_lengths + task.dpo.rejected_lengths
    else:
        rl = task.rl
        if e.fsdp != 1 or e.pp != 1:
            raise ValueError('RL currently uses shared serial roles with PP=FSDP=1; TP/EP/DP are supported')
        if rl.prompts_per_step * rl.samples_per_prompt % (w.batch_size * w.gradient_accumulation):
            raise ValueError('RL samples per DP replica must divide into complete batch_size * gradient_accumulation update batches')
        prompts, responses = rl_lengths(rl.model_dump())
        lengths = [p + r for p, r in zip(prompts, responses)]
        # Each optional model must be executable with the actual shared layout.
        from .config import Config
        for model in (rl.reward_model, rl.critic_model):
            if model:
                data = config.model_dump()
                data['workload']['training'] = None
                data['model'] = model.model_dump()
                Config.model_validate(data)
    if any(n < 1 or n > w.sequence_length for n in lengths):
        raise ValueError('All training sequence lengths must be positive and <= workload.sequence_length (RL includes prompt + response)')


def _counts(valid, processed, supervised, repeats, dp):
    return {'processed_tokens': sum(processed) * repeats * dp,
            'effective_tokens': sum(valid) * repeats * dp,
            'supervised_tokens': supervised * repeats * dp}


def _lengths(config, valid):
    return (list(valid) if config['workload']['training']['data']['batching'] == 'packed'
            else [config['workload']['sequence_length']] * len(valid))


def _aux(builder, name, rank, tokens, deps, kernel, flops, memory_bytes, workspace=0,
         *, phase='loss', lengths=None):
    """An explicit scalar operator, eligible for the usual trace/profile matching."""
    from .kernel_workload import kernel_workload_shape
    allocations = ()
    if workspace:
        allocations = (builder.allocation(name + '.workspace', rank, workspace, 'TrainingObjective',
                                          'operator_workspace', dtype='fp32'),)
    return builder.add(Event(name, tuple(deps), {builder.scalar(rank): flops, rank + '.hbm': memory_bytes},
        slots=(rank + '.compute_stream',), allocs=allocations, releases=tuple(a.id for a in allocations),
        metadata={'phase': phase, 'kernel': kernel, 'component': 'TrainingObjective', 'flops': flops,
                  'tokens': tokens, 'model_shape_signature': builder.model_shape_signature,
                  'tensor_parallel_degree': builder.tp, 'expert_parallel_degree': builder.ep,
                  'sequence_parallel_degree': 1,
                  'workload_shape': kernel_workload_shape(phase=phase if phase in ('prefill', 'decode') else 'forward', tokens=tokens,
                      lengths=builder.train_lengths if lengths is None else lengths,
                      training=builder.training, ep=builder.ep)}))


def _objective(builder, kind, extra_flops_per_token=0):
    """Full-position log-softmax cost; masking never scales the backbone backward."""
    def hook(prefix, stage, micro, deps):
        tokens = sum(builder.train_lengths) / builder.ep
        vocab = builder.a.get('output_head_dim') or builder.a['vocab_size']
        ends = []
        for rank, coord in builder.coords.items():
            if coord[1] != stage:
                continue
            logits = tokens * vocab / (1 if vocab == 1 else builder.tp)
            ends.append(_aux(builder, prefix + '.' + rank + '.' + kind, rank, tokens, deps, kind,
                5 * logits + extra_flops_per_token * tokens,
                2 * logits * DTYPE_BYTES[builder.p['activation']] + tokens * 8,
                workspace=tokens * 8))
        if builder.tp > 1:
            # Vocabulary-sharded max and normalization sum are separate reductions.
            for reduction in (('max', 'sum') if vocab > 1 else ('value_sum',)):
                collective_ends = []
                for index, ranks in enumerate(builder.groups(stage, 'tp')):
                    collective_ends += builder.collective(prefix + f'.{kind}.{reduction}{index}',
                        'all_reduce', ranks, tokens * 4, ends, 'TrainingObjective', builder.m['layers'] - 1,
                        phase='loss')
                ends = collective_ends
        return ends
    return hook


def _training_graph(config, topology, valid, *, frozen=False, model=None, accumulation=None,
                    objective='cross_entropy', extra_flops_per_token=0):
    from .structured import Builder
    data = deepcopy(config)
    data['workload'].update(training_steps=1, batch_size=len(valid))
    if accumulation is not None:
        data['workload']['gradient_accumulation'] = accumulation
    if model is not None:
        data['model'] = deepcopy(model)
    if frozen:
        data['execution']['fsdp'] = 1
        data['execution']['features'].update(optimizer_offload=False, activation_checkpointing=False)
    builder = Builder(data, topology)
    builder.forward_only = frozen
    builder.train_lengths = _lengths(config, valid)
    builder._training_loss = _objective(builder, objective, extra_flops_per_token)
    graph = builder.training_graph()
    builder.annotate_collective_allocations()
    return graph


def _rollout_graph(config, topology, prompts, responses):
    from .structured import Builder
    data = deepcopy(config)
    data['workload'].update(scenario='inference', training=None, batch_size=len(prompts),
                            input_lengths=prompts, output_lengths=responses)
    data['execution']['features'].update(optimizer_offload=False, activation_checkpointing=False)
    builder = Builder(data, topology)

    graph = builder.inference()
    outputs = {event.id: event for event in graph.events if event.metadata.get('component') == 'Output'}
    # Add sampling after each token's logits, before the next autoregressive step.
    for replica, sequences in graph.metadata['sequence_token_completion_events'].items():
        counts = defaultdict(int)
        for ids in sequences.values():
            for eid in ids:
                counts[eid] += 1
        barriers = {e.id: e for e in graph.events if e.id in counts}
        for eid in counts:
            barrier, ends = barriers[eid], []
            held_logits = []
            for rank, coord in builder.coords.items():
                if coord[0] != int(replica):
                    continue
                # Request ownership follows the inference EP assignment, including ragged tails.
                step = int(eid.rsplit('token', 1)[1])
                local = sum(i % builder.ep == coord[2] and responses[i] > step for i in range(len(responses)))
                if not local:
                    continue
                logits = local * builder.a['vocab_size'] / builder.tp
                output = outputs[f'infer.d{replica}.s{step}.l{builder.m["layers"] - 1}.{rank}.Output']
                held = {a.id for a in output.allocs if a.metadata.get('component') == 'Output' and a.id in output.releases}
                output.releases = tuple(aid for aid in output.releases if aid not in held)
                held_logits.extend(sorted(held))
                ends.append(_aux(builder, eid + '.' + rank + '.sampling', rank, local, barrier.deps,
                    'rollout_sampling', 5 * logits, 2 * logits * DTYPE_BYTES[builder.p['activation']], local * 8,
                    phase='prefill' if step == 0 else 'decode',
                    lengths=[prompts[i] + step for i in range(len(responses))
                             if i % builder.ep == coord[2] and responses[i] > step]))
            if builder.tp > 1:
                for reduction in ('max', 'sum'):
                    reduced = []
                    for index, ranks in enumerate(builder.groups(0, 'tp')):
                        coord = builder.coords[ranks[0]]
                        if coord[0] != int(replica):
                            continue
                        local = sum(i % builder.ep == coord[2] and responses[i] > step for i in range(len(responses)))
                        if local:
                            reduced += builder.collective(eid + f'.sampling.{reduction}{index}', 'all_reduce',
                                ranks, local * 4, ends, 'TrainingObjective', builder.m['layers'] - 1,
                                phase='prefill' if step == 0 else 'decode')
                    ends = reduced or ends
            barrier.deps = tuple(ends) or barrier.deps
            barrier.releases += tuple(held_logits)
    builder.annotate_collective_allocations()
    return graph


class _Schedule:
    """Compose serialized stages on one physical pool, sharing persistent role states."""
    def __init__(self, config, topology):
        self.config, self.topology = config, topology
        self.graph = Graph()
        topology.populate(self.graph)
        self.residents = {}
        self.previous = None
        self.stages = []
        self.role_logical = {}
        self.serial = 0

    def append(self, graph, stage, role, iteration, *, update=False):
        prefix = f'task.i{iteration}.{stage}.{self.serial}'
        self.serial += 1
        ids = {event.id: prefix + '.' + event.id for event in graph.events}
        allocations = {}
        for allocation in graph.initial_allocations:
            meta = allocation.metadata
            owner = 'shared_system' if meta.get('component') == 'System' else role
            key = (owner, allocation.pool, allocation.category, meta.get('tensor_name'), meta.get('layer'), allocation.dtype)
            if key in self.residents:
                existing = self.residents[key]
                if existing.size_bytes != allocation.size_bytes:
                    raise ValueError(f'Incompatible shared role state: {key}')
            else:
                existing = replace(allocation, id=f'task.resident.{len(self.residents)}',
                    metadata={**meta, 'training_role': owner})
                self.residents[key] = existing
                self.graph.initial_allocations.append(existing)
            allocations[allocation.id] = existing.id
        for event in graph.events:
            for allocation in event.allocs:
                allocations[allocation.id] = prefix + '.' + allocation.id
        self.graph.resources.update(graph.resources)
        self.graph.memory_caps.update(graph.memory_caps)
        start = prefix + '.start'
        self.graph.events.append(Event(start, (self.previous,) if self.previous else (),
                                       metadata={'training_stage': stage, 'training_role': role}))
        created, released = set(), set()
        for event in graph.events:
            metadata = {**event.metadata, 'training_task': self.config['workload']['training']['kind'],
                        'training_stage': stage, 'training_role': role, 'training_update': update,
                        'training_iteration': iteration}
            self.graph.events.append(replace(event, id=ids[event.id],
                deps=tuple(ids[d] for d in event.deps) or (start,),
                allocs=tuple(replace(a, id=allocations[a.id], metadata={**a.metadata, 'training_role': role}) for a in event.allocs),
                releases=tuple(allocations[a] for a in event.releases), metadata=metadata))
            created.update(a.id for a in event.allocs)
            released.update(event.releases)
        used = {dep for event in graph.events for dep in event.deps}
        terminals = [ids[e.id] for e in graph.events if e.id not in used]
        end = prefix + '.done'
        # Inference deliberately retains fixed-batch KV until batch exit. Release it here.
        self.graph.events.append(Event(end, tuple(terminals) or (start,),
            releases=tuple(allocations[a] for a in sorted(created - released)),
            metadata={'training_stage': stage, 'training_role': role}))
        self.previous = end
        self.stages.append({'stage': stage, 'role': role, 'iteration': iteration,
                            'start_event': start, 'end_event': end})
        # The actor's training ledger supersedes its rollout-only weight ledger.
        if role not in self.role_logical or update:
            self.role_logical[role] = graph.metadata.get('logical_memory_bytes', {})
        for key in ('component_inventory', 'assumptions', 'unsupported_production_features', 'sequence_parallel'):
            if key not in self.graph.metadata and key in graph.metadata:
                self.graph.metadata[key] = deepcopy(graph.metadata[key])
        return end


def build_training_task(config, topology):
    task, w, e = config['workload']['training'], config['workload'], config['execution']
    schedule = _Schedule(config, topology)
    counts = defaultdict(int)
    iteration_ends = []
    if task['kind'] in ('pretrain', 'sft', 'dpo'):
        if task['kind'] == 'dpo':
            valid = task['dpo']['chosen_lengths'] + task['dpo']['rejected_lengths']
        else:
            valid = task['data']['sequence_lengths'] or [w['sequence_length']] * w['batch_size']
        supervised = (sum(task['sft']['supervised_tokens'])
                      if task['kind'] == 'sft' and task['sft']['loss_mask'] == 'response' else sum(valid))
        actor = _training_graph(config, topology, valid, objective='dpo_loss' if task['kind'] == 'dpo' else 'cross_entropy',
                                extra_flops_per_token=10 * len(valid) / sum(_lengths(config, valid)) if task['kind'] == 'dpo' else 0)
        reference = (_training_graph(config, topology, valid, frozen=True, objective='reference_log_probs')
                     if task['kind'] == 'dpo' and task['dpo']['reference_log_probs'] == 'online' else None)
        for iteration in range(w['training_steps']):
            preference_buffers = []
            if task['kind'] == 'dpo':
                from .structured import Builder
                builder = Builder(config, topology)
                for rank, coord in builder.coords.items():
                    if coord[1] == builder.pp - 1:
                        preference_buffers.append(builder.allocation(f'preference.i{iteration}.{rank}', rank,
                            len(valid) * w['gradient_accumulation'] * 4 / e['ep'], 'Preference', 'reference_log_probs',
                            dtype='fp32', training_role='reference'))
                start = f'task.i{iteration}.preference.start'
                schedule.graph.events.append(Event(start, (schedule.previous,) if schedule.previous else (),
                                                    allocs=tuple(preference_buffers)))
                schedule.previous = start
            if reference:
                schedule.append(reference, 'reference', 'reference', iteration)
            schedule.append(actor, 'actor_update', 'actor', iteration, update=True)
            if preference_buffers:
                end = f'task.i{iteration}.preference.done'
                schedule.graph.events.append(Event(end, (schedule.previous,), releases=tuple(a.id for a in preference_buffers)))
                schedule.previous = end
            iteration_ends.append(schedule.previous)
        counts.update(_counts(valid, _lengths(config, valid), supervised,
                              w['gradient_accumulation'] * w['training_steps'], e['dp']))
        counts['optimizer_steps'] = w['training_steps']
        counts['samples'] = len(valid) * w['gradient_accumulation'] * w['training_steps'] * e['dp']
        if task['kind'] == 'dpo':
            counts['preference_pairs'] = counts['samples'] // 2
            # Pairwise labels are not supervised token labels.
            counts['supervised_tokens'] = None
    else:
        _build_rl(schedule, config, topology, counts, iteration_ends)
    graph = schedule.graph
    graph.metadata.update(scenario='training', step_end_events=iteration_ends,
        training_steps=w['training_steps'], training_task={
            'kind': task['kind'], 'parameters': deepcopy(task), 'counts': dict(counts),
            'stages': schedule.stages, 'role_logical_memory_bytes': schedule.role_logical,
            'schedule': 'shared_pool_serial_roles' if task['kind'] in ('rl', 'dpo') else 'training_steps',
            'throughput_basis': 'actor processed training tokens, including padding and repeated updates; excludes rollout/scoring/critic tokens',
            'limitations': ['Analytical forward/backward proxy; no optimization quality or convergence prediction.',
                'Roles use the same physical TP/EP/DP layout and precision; all persistent role states coexist.',
                'No asynchronous rollout, separate actor/rollout pools, LoRA/QLoRA, or partial-layer freezing.',
                'Packing uses independent causal segments; no cross-document attention or prefix-cache reuse.',
                'Full-position logits and log-softmax are modeled; loss masking does not prune backbone or output GEMMs.']})
    totals = defaultdict(float)
    for values in schedule.role_logical.values():
        for key, value in values.items():
            totals[key] += value
    graph.metadata['logical_memory_bytes'] = dict(totals)
    if task['kind'] in ('rl', 'dpo'):
        # Actor-only component totals would misrepresent frozen/critic copies.
        graph.metadata.pop('logical_component_state_bytes', None)
    graph.metadata.setdefault('assumptions', []).extend(graph.metadata['training_task']['limitations'])
    return graph


def _build_rl(schedule, config, topology, counts, iteration_ends):
    from .structured import Builder
    task, w, e = config['workload']['training'], config['workload'], config['execution']
    rl = task['rl']
    prompts, responses = rl_lengths(rl)
    valid = [p + r for p, r in zip(prompts, responses)]
    micro, accumulation = w['batch_size'], w['gradient_accumulation']
    for iteration in range(w['training_steps']):
        # Experience is live from collection through every actor/critic update.
        builder = Builder(config, topology)
        buffers = []
        for rank in builder.coords:
            size = sum(valid) / e['ep'] * (rl['experience_bytes_per_token'] + (4 if rl['algorithm'] == 'ppo' else 0))
            buffers.append(builder.allocation(f'experience.i{iteration}.{rank}', rank, size,
                'Experience', 'experience_buffer', dtype=None, training_role='experience'))
        allocation_event = f'task.i{iteration}.experience.start'
        schedule.graph.events.append(Event(allocation_event, (schedule.previous,) if schedule.previous else (), allocs=tuple(buffers)))
        schedule.previous = allocation_event
        for offset in range(0, len(valid), rl['rollout_batch_size']):
            section = slice(offset, offset + rl['rollout_batch_size'])
            schedule.append(_rollout_graph(config, topology, prompts[section], responses[section]),
                            'rollout', 'actor', iteration)
        for offset in range(0, len(valid), micro):
            lengths = valid[offset:offset + micro]
            if rl['old_log_probs'] == 'recompute':
                schedule.append(_training_graph(config, topology, lengths, frozen=True, accumulation=1,
                    objective='old_log_probs'), 'old_log_probs', 'actor', iteration)
            if rl['reference_policy']:
                schedule.append(_training_graph(config, topology, lengths, frozen=True, accumulation=1,
                    objective='reference_log_probs'), 'reference', 'reference', iteration)
            if rl['reward'] == 'model':
                schedule.append(_training_graph(config, topology, lengths, frozen=True, model=rl['reward_model'],
                    accumulation=1, objective='reward_score'), 'reward_model', 'reward', iteration)
            if rl['algorithm'] == 'ppo':
                schedule.append(_training_graph(config, topology, lengths, frozen=True, model=rl['critic_model'],
                    accumulation=1, objective='value_score'), 'value', 'critic', iteration)
        if rl['reward'] == 'rule':
            graph = Graph(events=[Event('rule_reward', startup_s=ceil(len(valid) * e['dp'] / rl['reward_workers']) * rl['rule_reward_latency_s'],
                                       metadata={'phase': 'reward', 'kernel': 'external_rule_reward'})])
            schedule.append(graph, 'rule_reward', 'environment', iteration)
        builder = Builder(config, topology)
        builder.train_lengths = valid
        for rank in builder.coords:
            tokens = sum(responses) / e['ep']
            _aux(builder, rank + '.advantages', rank, tokens, (),
                 'group_advantages' if rl['algorithm'] == 'grpo' else 'gae',
                 tokens * rl['advantage_flops_per_token'], tokens * 24, tokens * 4)
        if e['ep'] > 1 and rl['algorithm'] == 'grpo':
            deps = [event.id for event in builder.g.events]
            for index, ranks in enumerate(builder.groups(0, 'ep')):
                builder.collective(f'advantages.group_stats{index}', 'all_reduce', ranks,
                    rl['prompts_per_step'] * 3 * 4, deps, 'TrainingObjective', 0, phase='loss')
        schedule.append(builder.g, 'advantage', 'experience', iteration)
        # Different microbatches may have different shapes. Preserve one optimizer
        # boundary per accumulation group by using a per-microbatch shape schedule.
        for epoch in range(rl['update_epochs']):
            for offset in range(0, len(valid), micro * accumulation):
                groups = [valid[i:i + micro] for i in range(offset, offset + micro * accumulation, micro)]
                for role, model, objective in [('actor', None, 'policy_loss')] + (
                        [('critic', rl['critic_model'], 'value_loss')] if rl['algorithm'] == 'ppo' else []):
                    graph = _variable_update(config, topology, groups, model=model, objective=objective,
                                             extra=rl['policy_loss_flops_per_token'] if role == 'actor' else 4)
                    schedule.append(graph, role + '_update', role, iteration, update=True)
                counted = valid[offset:offset + micro * accumulation]
                for key, value in _counts(counted, _lengths(config, counted),
                        sum(responses[offset:offset + micro * accumulation]), 1, e['dp']).items():
                    counts[key] += value
                counts['optimizer_steps'] += 1
                counts['critic_optimizer_steps'] += int(rl['algorithm'] == 'ppo')
        end = f'task.i{iteration}.done'
        schedule.graph.events.append(Event(end, (schedule.previous,), releases=tuple(a.id for a in buffers)))
        schedule.previous = end
        iteration_ends.append(end)
        counts['rollout_sequences'] += len(valid) * e['dp']
        counts['rollout_prompt_tokens'] += sum(prompts) * e['dp']
        counts['rollout_generated_tokens'] += sum(responses) * e['dp']
        counts['samples'] += len(valid) * rl['update_epochs'] * e['dp']


def _variable_update(config, topology, groups, *, model, objective, extra):
    """Use actual lengths for every microbatch without adding optimizer steps."""
    from .structured import Builder
    data = deepcopy(config)
    data['workload'].update(training_steps=1, batch_size=len(groups[0]), gradient_accumulation=len(groups))
    if model is not None:
        data['model'] = deepcopy(model)
    builder = Builder(data, topology)
    builder.training_lengths_by_micro = [_lengths(config, values) for values in groups]
    builder._training_loss = _objective(builder, objective, extra)
    graph = builder.training_graph()
    builder.annotate_collective_allocations()
    return graph


def training_task_metrics(config, graph, simulation):
    task = graph.metadata.get('training_task')
    if not task:
        return None
    duration = simulation['duration_s']
    counts = dict(task['counts'])
    wanted = {stage[key] for stage in task['stages'] for key in ('start_event', 'end_event')}
    times = {row['id']: row['end_s'] for row in simulation['trace'] if row['id'] in wanted}
    stages, by_stage = [], defaultdict(float)
    for stage in task['stages']:
        elapsed = times[stage['end_event']] - times[stage['start_event']]
        stages.append({**stage, 'duration_s': elapsed})
        by_stage[stage['stage']] += elapsed
    result = {'kind': task['kind'], **counts, 'stage_times_s': dict(by_stage), 'stages': stages,
              'throughput_basis': task['throughput_basis'],
              'effective_tokens_s': counts['effective_tokens'] / duration if duration else 0,
              'supervised_tokens_s': counts['supervised_tokens'] / duration if duration and counts['supervised_tokens'] is not None else None}
    if task['kind'] == 'rl':
        result.update(iterations=config.workload.training_steps,
                      rollout_tokens_s=counts['rollout_generated_tokens'] / duration if duration else 0,
                      rollout_sequences_s=counts['rollout_sequences'] / duration if duration else 0)
    budget = task['parameters'].get('pretrain')
    if budget and budget['dataset_tokens']:
        tokens_per_step = counts['effective_tokens'] / config.workload.training_steps
        steps = ceil(budget['dataset_tokens'] * budget['epochs'] / tokens_per_step)
        result['budget_projection'] = {'optimizer_steps': steps, 'duration_s': steps * duration / config.workload.training_steps,
            'basis': 'Extrapolation of mean simulated step; final batch rounded up, no convergence or data-loader model.'}
    return result

