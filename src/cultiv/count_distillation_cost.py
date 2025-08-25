import argparse
import collections
import math
import numpy as np

import stim

import gen
import cultiv


def circuit_to_layers(circuit: stim.Circuit) -> list[stim.Circuit]:
    cur_layer = stim.Circuit()
    prev_layers = []
    saw_two_qubit_gate = False
    saw_measurement = False
    for inst in circuit.flattened():
        data = stim.GateData(inst.name)
        if inst.name == 'QUBIT_COORDS':
            continue
        elif data.is_two_qubit_gate:
            if saw_measurement:
                saw_measurement = False
                prev_layers.append(cur_layer)
                cur_layer = stim.Circuit()
            saw_two_qubit_gate = True
            cur_layer.append(inst)
        elif data.produces_measurements:
            cur_layer.append(inst)
            saw_measurement = True
        elif data.is_reset or data.produces_measurements:
            if saw_two_qubit_gate:
                saw_two_qubit_gate = False
                saw_measurement = False
                prev_layers.append(cur_layer)
                cur_layer = stim.Circuit()
            cur_layer.append(inst)
        else:
            cur_layer.append(inst)
    if len(cur_layer):
        prev_layers.append(cur_layer)
    return prev_layers


def sample_times(circuit: stim.Circuit, shots: int) -> tuple[collections.Counter, list[int]]:
    sim = stim.FlipSimulator(batch_size=1024, num_qubits=circuit.num_qubits)
    layers = circuit_to_layers(circuit)

    qubit_counts = []
    used_qubits = set()
    for layer in layers:
        for inst in layer:
            if inst.name in ['R', 'RX']:
                for t in inst.targets_copy():
                    used_qubits.add(t.qubit_value)
        qubit_counts.append(len(used_qubits))
    survivors = np.zeros(1024, dtype=np.bool_)
    postselected_detectors = set()
    for det, coord in circuit.get_detector_coordinates().items():
        if len(coord) == 3 or coord[-1] == -9 or coord[4] == 0 or coord[4] == 4:
            postselected_detectors.add(det)
    counts = collections.Counter()

    shots_left = shots
    while shots_left > 0:
        sim.clear()
        cur_det = 0
        tick = 0
        survivors[:] = True
        for layer in layers:
            tick += 1
            sim.do(layer)
            if cur_det < sim.num_detectors:
                while cur_det < sim.num_detectors:
                    if cur_det in postselected_detectors:
                        fired = sim.get_detector_flips(detector_index=cur_det)
                        survivors &= ~fired
                    cur_det += 1
            counts[tick] += np.count_nonzero(survivors)

        counts[0] += 1024
        shots_left -= 1024

    return counts, qubit_counts


def desc(n: float) -> str:
    power = math.floor(math.log10(n))
    while 10**(power + 1) <= n:
        power += 1
    base = round(n / 10**power * 10) / 10
    base = str(base).ljust(3, '0')
    return fr'${base} \cdot 10^{{{power}}}$'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--end-to-end-success-rate', type=float, required=True)
    args = parser.parse_args()

    print('  end-to-end-success-rate = {}'.format(args.end_to_end_success_rate))

    end_to_end_success_rate: float = args.end_to_end_success_rate

    dcolor = 3
    circuit = cultiv.make_end2end_cultivation_circuit(
        dcolor=dcolor,
        dsurface=15,
        basis='Y',
        r_growing=dcolor,
        r_end=10,
        inject_style='unitary',
    )
    circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
    num_shots = 1024*100
    ts, qs = sample_times(circuit, num_shots)
    assert len(ts) == len(qs) + 1

    num_rounds = len(qs)
    ts[num_rounds] = round(num_shots * args.end_to_end_success_rate)
    print()

    cost = 0

    for r in range(num_rounds):
        round_success_rate = ts[r + 1] / ts[r] 
        assert 0 <= round_success_rate
        assert round_success_rate <= 1

        cost = (cost + qs[r]) / round_success_rate

    print('expected cost = {}'.format(cost))


if __name__ == '__main__':
    main()
