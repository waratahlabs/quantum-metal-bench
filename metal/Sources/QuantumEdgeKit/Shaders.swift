import Foundation

public let metalSrc: String = """
#include <metal_stdlib>
using namespace metal;

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

kernel void apply_single_qubit_gate(
    device float2*       sv       [[ buffer(0) ]],
    constant float2*     U        [[ buffer(1) ]],
    constant uint&       target   [[ buffer(2) ]],
    constant uint&       n_qubits [[ buffer(3) ]],
    uint                 gid      [[ thread_position_in_grid ]])
{
    const uint n_half = 1u << (n_qubits - 1u);
    if (gid >= n_half) return;
    const uint stride = 1u << target;
    const uint mask   = stride - 1u;
    const uint i0     = ((gid & ~mask) << 1u) | (gid & mask);
    const uint i1     = i0 | stride;
    const float2 a = sv[i0], b = sv[i1];
    sv[i0] = cmul(U[0], a) + cmul(U[1], b);
    sv[i1] = cmul(U[2], a) + cmul(U[3], b);
}

kernel void apply_two_qubit_gate(
    device float2*       sv       [[ buffer(0) ]],
    constant float2*     U        [[ buffer(1) ]],
    constant uint&       q_lo     [[ buffer(2) ]],
    constant uint&       q_hi     [[ buffer(3) ]],
    constant uint&       n_qubits [[ buffer(4) ]],
    uint                 gid      [[ thread_position_in_grid ]])
{
    const uint quarter = 1u << (n_qubits - 2u);
    if (gid >= quarter) return;
    uint lo_mask = (1u << q_lo) - 1u;
    uint mid = ((gid & ~lo_mask) << 1u) | (gid & lo_mask);
    uint hi_mask = (1u << q_hi) - 1u;
    uint base = ((mid & ~hi_mask) << 1u) | (mid & hi_mask);

    const uint i00 = base;
    const uint i01 = base | (1u << q_lo);
    const uint i10 = base | (1u << q_hi);
    const uint i11 = base | (1u << q_lo) | (1u << q_hi);

    const float2 a = sv[i00], b = sv[i01], c = sv[i10], d = sv[i11];
    sv[i00] = cmul(U[0],a)  + cmul(U[1],b)  + cmul(U[2],c)  + cmul(U[3],d);
    sv[i01] = cmul(U[4],a)  + cmul(U[5],b)  + cmul(U[6],c)  + cmul(U[7],d);
    sv[i10] = cmul(U[8],a)  + cmul(U[9],b)  + cmul(U[10],c) + cmul(U[11],d);
    sv[i11] = cmul(U[12],a) + cmul(U[13],b) + cmul(U[14],c) + cmul(U[15],d);
}
"""
