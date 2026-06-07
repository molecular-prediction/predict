import os
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from gnr_types import BenzeneHex, GlobalCutPlan
from gnr_graph import (
    classify_edge_type,
    _cluster_1d,
    _vertical_stack_heights,
    _minimal_sequence_period,
)


@dataclass
class MonomerGenerationResult:
    is_valid: bool = False
    failure_reason: str = ""
    raw_smiles: List[str] = field(default_factory=list)
    capped_smiles: List[str] = field(default_factory=list)
    raw_files: List[str] = field(default_factory=list)
    capped_files: List[str] = field(default_factory=list)
    monomer_images: List[str] = field(default_factory=list)

def extract_submol_from_hexes(original_mol, hexes_subset: List[BenzeneHex]):
    """提取原始未封端分子骨架"""
    atom_indices = set()
    for h in hexes_subset:
        for aid in h.atom_indices:
            atom_indices.add(aid)

    if not atom_indices: return None

    rw_mol = Chem.RWMol()
    old_to_new_map = {}
    sorted_old_indices = sorted(list(atom_indices))

    for old_idx in sorted_old_indices:
        atom = original_mol.GetAtomWithIdx(old_idx)
        new_idx = rw_mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        rw_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
        old_to_new_map[old_idx] = new_idx

    for old_idx in sorted_old_indices:
        orig_atom = original_mol.GetAtomWithIdx(old_idx)
        for neighbor in orig_atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in atom_indices and n_idx > old_idx:
                bond = original_mol.GetBondBetweenAtoms(old_idx, n_idx)
                if bond:
                    rw_mol.AddBond(old_to_new_map[old_idx], old_to_new_map[n_idx], bond.GetBondType())

    return rw_mol

def get_atom_to_hexes_map(all_hexes: List[BenzeneHex]) -> Dict[int, List[BenzeneHex]]:
    """建立 原子ID 到 苯环对象 的反向映射"""
    mapping = {}
    for h in all_hexes:
        for idx in h.atom_indices:
            mapping.setdefault(idx, []).append(h)
    return mapping


def _count_aromatic_rings(mol: Chem.Mol) -> int:
    ring_info = mol.GetRingInfo()
    count = 0
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            count += 1
    return count


def _count_atoms_by_symbol(mol: Chem.Mol, symbol: str) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)


def _count_terminal_aliphatic_carbons(mol: Chem.Mol) -> int:
    return sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C" and not atom.GetIsAromatic() and atom.GetDegree() == 1
    )


def _accept_capped_monomer(mol: Chem.Mol, edge_type: str) -> bool:
    """按 edge_type 决定封端产物是否合格（数量过滤）。

    - zigzag：Br==2 且末端脂肪碳(甲基)==2（保 7ac/4.smi 基线）。
    - armchair：Br==2 且甲基 in {0,2}（竖直 fused 堆上下端通常无固定边断键→0 甲基）。
      至少 2 个 Br 保证周期方向可继续聚合。
    """
    if _count_atoms_by_symbol(mol, "Br") != 2:
        return False
    methyl = _count_terminal_aliphatic_carbons(mol)
    if edge_type == "armchair":
        return methyl in (0, 2)
    return methyl == 2


def _hexes_form_connected_fused_component(window_hexes: List[BenzeneHex]) -> bool:
    if not window_hexes:
        return False
    if len(window_hexes) == 1:
        return True

    id_to_index = {h.id: idx for idx, h in enumerate(window_hexes)}
    adjacency = {h.id: set() for h in window_hexes}
    for i, h1 in enumerate(window_hexes):
        atoms1 = set(h1.atom_indices)
        for h2 in window_hexes[i + 1:]:
            shared_atoms = atoms1.intersection(h2.atom_indices)
            if len(shared_atoms) >= 2:
                adjacency[h1.id].add(h2.id)
                adjacency[h2.id].add(h1.id)

    seen = {window_hexes[0].id}
    stack = [window_hexes[0].id]
    while stack:
        hid = stack.pop()
        for nid in adjacency[hid]:
            if nid not in seen:
                seen.add(nid)
                stack.append(nid)
    return len(seen) == len(id_to_index)

def _collect_cut_boundary(
    original_mol,
    all_hexes: List[BenzeneHex],
    cutting_edges: Dict[int, List[Tuple[int, int]]],
) -> Tuple[set, set]:
    id_to_hex = {h.id: h for h in all_hexes}
    cut_hex_ids = set()
    cut_atom_bonds = set()

    for path in cutting_edges.values():
        for u, v in path:
            cut_hex_ids.update([u, v])
            h1 = id_to_hex.get(u)
            h2 = id_to_hex.get(v)
            if not h1 or not h2:
                continue
            shared_atoms = sorted(set(h1.atom_indices).intersection(h2.atom_indices))
            if len(shared_atoms) != 2:
                continue
            a1, a2 = shared_atoms
            if original_mol.GetBondBetweenAtoms(a1, a2):
                cut_atom_bonds.add(frozenset((a1, a2)))

    return cut_hex_ids, cut_atom_bonds


def extract_all_capped_monomers(
    original_mol,
    window_hexes: List[BenzeneHex],
    all_hexes: List[BenzeneHex],
    cut_atom_bonds: set,
    top_exit_direction: str = "",
) -> List[Chem.Mol]:
    monomer_atom_indices = set(aid for h in window_hexes for aid in h.atom_indices)
    if not monomer_atom_indices:
        return []

    base_rw_mol = Chem.RWMol()
    old_to_new_map = {}
    sorted_old_indices = sorted(list(monomer_atom_indices))

    for old_idx in sorted_old_indices:
        atom = original_mol.GetAtomWithIdx(old_idx)
        new_idx = base_rw_mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        base_rw_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
        old_to_new_map[old_idx] = new_idx

    for old_idx in sorted_old_indices:
        orig_atom = original_mol.GetAtomWithIdx(old_idx)
        for neighbor in orig_atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in monomer_atom_indices and n_idx > old_idx:
                bond = original_mol.GetBondBetweenAtoms(old_idx, n_idx)
                if bond:
                    base_rw_mol.AddBond(old_to_new_map[old_idx], old_to_new_map[n_idx], bond.GetBondType())

    conf = original_mol.GetConformer()
    broken_bonds = []
    for old_idx in sorted_old_indices:
        orig_atom = original_mol.GetAtomWithIdx(old_idx)
        for neighbor in orig_atom.GetNeighbors():
            if neighbor.GetIdx() not in monomer_atom_indices:
                broken_bonds.append(old_idx)
                break

    if not broken_bonds:
        return []

    if len(window_hexes) == 2:
        h1, h2 = window_hexes[0], window_hexes[1]
        if h1.col > h2.col or (h1.col == h2.col and h1.cx > h2.cx):
            h1, h2 = h2, h1

        bridgeheads = set(h1.atom_indices).intersection(set(h2.atom_indices))
        if len(bridgeheads) != 2:
            return []

        vec_x = (h2.cx - h1.cx, h2.cy - h1.cy)
        length_x = math.hypot(vec_x[0], vec_x[1]) or 1.0
        ux = (vec_x[0] / length_x, vec_x[1] / length_x)
        uy = (-ux[1], ux[0])
        mcx = (h1.cx + h2.cx) / 2
        mcy = (h1.cy + h2.cy) / 2

        def get_local_y(u):
            pos = conf.GetAtomPosition(u)
            return (pos.x - mcx) * uy[0] + (pos.y - mcy) * uy[1]

        left_bonds = [u for u in broken_bonds if u in h1.atom_indices and u not in h2.atom_indices]
        right_bonds = [u for u in broken_bonds if u in h2.atom_indices and u not in h1.atom_indices]

        def categorize_and_sort(bonds):
            alphas, betas = [], []
            for u in bonds:
                if any(n.GetIdx() in bridgeheads for n in original_mol.GetAtomWithIdx(u).GetNeighbors()):
                    alphas.append(u)
                else:
                    betas.append(u)
            alphas.sort(key=get_local_y, reverse=True)
            betas.sort(key=get_local_y, reverse=True)
            return alphas, betas

        left_alphas, left_betas = categorize_and_sort(left_bonds)
        right_alphas, right_betas = categorize_and_sort(right_bonds)

        left_choices = []
        if left_alphas:
            br_bond_left = left_alphas[-1]
            if len(left_alphas) >= 2:
                left_choices.append({br_bond_left: "Br", left_alphas[0]: "C"})
            if len(left_betas) >= 2:
                left_choices.append({br_bond_left: "Br", left_betas[-1]: "C"})
            elif len(left_betas) == 1:
                left_choices.append({br_bond_left: "Br", left_betas[0]: "C"})
            if not left_choices:
                left_choices.append({br_bond_left: "Br"})
        else:
            left_choices = [{}]

        right_choices = []
        if right_alphas:
            br_bond_right = right_alphas[0]
            if len(right_alphas) >= 2:
                right_choices.append({br_bond_right: "Br", right_alphas[-1]: "C"})
            if right_betas:
                right_choices.append({br_bond_right: "Br", right_betas[0]: "C"})
            if not right_choices:
                right_choices.append({br_bond_right: "Br"})
        else:
            right_choices = [{}]

        capped_mols = []
        for left_choice in left_choices:
            for right_choice in right_choices:
                rw_mol = Chem.RWMol(base_rw_mol)
                assignments = {}
                assignments.update(left_choice)
                assignments.update(right_choice)
                for atom_idx, cap_type in assignments.items():
                    cap = rw_mol.AddAtom(Chem.Atom(cap_type))
                    rw_mol.AddBond(old_to_new_map[atom_idx], cap, Chem.BondType.SINGLE)
                try:
                    Chem.SanitizeMol(rw_mol)
                    capped_mols.append(rw_mol.GetMol())
                except Exception:
                    pass
        return capped_mols

    def get_local_y(u):
        return conf.GetAtomPosition(u).y

    window_cols = [h.col for h in window_hexes]
    min_window_col = min(window_cols)
    max_window_col = max(window_cols)

    cut_bonds = set()
    boundary_bonds = set()
    external_neighbors = {}
    atom_to_cols = {}
    for h in all_hexes:
        for atom_idx in h.atom_indices:
            atom_to_cols.setdefault(atom_idx, set()).add(h.col)

    def get_periodic_side(u):
        neighbor_idx = external_neighbors.get(u)
        neighbor_cols = atom_to_cols.get(neighbor_idx, set())
        if neighbor_cols:
            if max(neighbor_cols) < min_window_col:
                return "left"
            if min(neighbor_cols) > max_window_col:
                return "right"
        return ""

    for u in broken_bonds:
        for neighbor in original_mol.GetAtomWithIdx(u).GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in monomer_atom_indices:
                continue
            external_neighbors[u] = n_idx
            bond_key = frozenset((u, n_idx))
            if bond_key in cut_atom_bonds:
                cut_bonds.add(u)
            else:
                boundary_bonds.add(u)

    if not cut_bonds:
        return []

    left_cut_bonds = [u for u in cut_bonds if get_periodic_side(u) == "left"]
    right_cut_bonds = [u for u in cut_bonds if get_periodic_side(u) == "right"]
    left_cut_bonds.sort(key=get_local_y, reverse=True)
    right_cut_bonds.sort(key=get_local_y, reverse=True)

    left_boundary_bonds = [u for u in boundary_bonds if get_periodic_side(u) == "left"]
    right_boundary_bonds = [u for u in boundary_bonds if get_periodic_side(u) == "right"]
    left_boundary_bonds.sort(key=get_local_y, reverse=True)
    right_boundary_bonds.sort(key=get_local_y, reverse=True)

    def coupling_pair_score(left_bond, right_bond):
        left_neighbor = external_neighbors.get(left_bond)
        right_neighbor = external_neighbors.get(right_bond)
        score = 0.0
        if right_neighbor is not None:
            score += abs(get_local_y(left_bond) - get_local_y(right_neighbor))
        if left_neighbor is not None:
            score += abs(get_local_y(right_bond) - get_local_y(left_neighbor))
        return score

    def pair_periodic_counterparts(left_bonds, right_bonds):
        candidates = sorted(
            (
                (coupling_pair_score(left, right), left, right)
                for left in left_bonds
                for right in right_bonds
            ),
            key=lambda item: item[0],
        )
        pairs = []
        used_left = set()
        used_right = set()
        for _score, left, right in candidates:
            if left in used_left or right in used_right:
                continue
            pairs.append((left, right))
            used_left.add(left)
            used_right.add(right)
        return pairs

    br_pairs = []
    if left_cut_bonds and right_cut_bonds:
        br_pairs.extend(pair_periodic_counterparts(left_cut_bonds, right_cut_bonds))
    elif left_cut_bonds and right_boundary_bonds:
        br_pairs.extend(pair_periodic_counterparts(left_cut_bonds, right_boundary_bonds))
    elif right_cut_bonds and left_boundary_bonds:
        br_pairs.extend(pair_periodic_counterparts(left_boundary_bonds, right_cut_bonds))
    br_pairs = list(dict.fromkeys(tuple(pair) for pair in br_pairs))

    if not br_pairs:
        return []

    capped_mols = []
    top_row = min(h.row for h in window_hexes)
    bottom_row = max(h.row for h in window_hexes)
    top_hexes = [h for h in window_hexes if h.row == top_row]
    bottom_hexes = [h for h in window_hexes if h.row == bottom_row]

    def is_top_second_row_atom(atom_idx):
        for h in top_hexes:
            if atom_idx not in h.atom_indices:
                continue
            ranked_atoms = sorted(
                h.atom_indices,
                key=lambda idx: conf.GetAtomPosition(idx).y,
                reverse=True,
            )
            if atom_idx in ranked_atoms[1:3]:
                return True
        return False

    def is_bottom_penultimate_row_atom(atom_idx):
        for h in bottom_hexes:
            if atom_idx not in h.atom_indices:
                continue
            ranked_atoms = sorted(
                h.atom_indices,
                key=lambda idx: conf.GetAtomPosition(idx).y,
            )
            if atom_idx in ranked_atoms[1:3]:
                return True
        return False

    top_carbon_bonds = sorted(
        [u for u in boundary_bonds if is_top_second_row_atom(u)],
        key=lambda u: conf.GetAtomPosition(u).x,
    )
    bottom_carbon_bonds = sorted(
        [u for u in boundary_bonds if is_bottom_penultimate_row_atom(u)],
        key=lambda u: conf.GetAtomPosition(u).x,
    )

    carbon_choices = []
    for top_bond in top_carbon_bonds:
        for bottom_bond in bottom_carbon_bonds:
            if top_bond != bottom_bond:
                carbon_choices.append((top_bond, bottom_bond))

    for br_bonds in br_pairs:
        for carbon_bonds in carbon_choices:
            if set(br_bonds).intersection(carbon_bonds):
                continue
            rw_mol = Chem.RWMol(base_rw_mol)
            for u in br_bonds:
                cap = rw_mol.AddAtom(Chem.Atom("Br"))
                rw_mol.AddBond(old_to_new_map[u], cap, Chem.BondType.SINGLE)
            for carbon_bond in carbon_bonds:
                cap = rw_mol.AddAtom(Chem.Atom("C"))
                rw_mol.AddBond(old_to_new_map[carbon_bond], cap, Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(rw_mol)
                capped_mols.append(rw_mol.GetMol())
            except Exception:
                pass
    return capped_mols

def _assign_vertical_stacks(all_hexes: List[BenzeneHex]) -> Tuple[Dict[int, int], int]:
    """把每个环分配到一个竖直堆（按 cx 聚类），返回 (hex_id->stack_index, 堆数)。

    armchair 蜂窝相邻堆错开半周期，mol_to_hex_grid 的 col 分桶会把它们合并，
    无法用于切出可 kekulize 的周期单元；这里按 cx 还原真实竖直堆。
    """
    if not all_hexes:
        return {}, 0
    avg_ring_size = sum(h.size for h in all_hexes) / len(all_hexes)
    centers = [sum(c) / len(c) for c in _cluster_1d([h.cx for h in all_hexes], avg_ring_size * 0.6)]
    stack_of: Dict[int, int] = {}
    for h in all_hexes:
        nearest = min(range(len(centers)), key=lambda i: abs(centers[i] - h.cx))
        stack_of[h.id] = nearest
    return stack_of, len(centers)


def _extract_armchair_capped_monomers(
    original_mol,
    window_hexes: List[BenzeneHex],
    window_stacks: set,
    atom_to_stacks: Dict[int, set],
) -> List[Chem.Mol]:
    """armchair 周期单元封端：在跨周期边界（左/右堆外）的 biaryl 断键上各封一个 Br。

    每个断键都封端，封什么由其外部邻居方向决定：
      - 周期方向（外部邻居堆在窗口左/右之外）→ Br，左右各取一个组合配对；
      - 固定边（外部邻居在窗口堆范围内，指向上/下边界外）→ 甲基(C)。
    armchair 竖直 fused 堆上下端通常是完整环边、带 H（无固定边断键），此时
    0 甲基，合法。保留能 SanitizeMol 的产物。
    """
    monomer_atoms = set(aid for h in window_hexes for aid in h.atom_indices)
    if not monomer_atoms:
        return []

    base_rw_mol = Chem.RWMol()
    old_to_new_map = {}
    sorted_old_indices = sorted(monomer_atoms)
    for old_idx in sorted_old_indices:
        atom = original_mol.GetAtomWithIdx(old_idx)
        new_idx = base_rw_mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        base_rw_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
        old_to_new_map[old_idx] = new_idx
    for old_idx in sorted_old_indices:
        orig_atom = original_mol.GetAtomWithIdx(old_idx)
        for neighbor in orig_atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in monomer_atoms and n_idx > old_idx:
                bond = original_mol.GetBondBetweenAtoms(old_idx, n_idx)
                if bond:
                    base_rw_mol.AddBond(old_to_new_map[old_idx], old_to_new_map[n_idx], bond.GetBondType())

    min_stack = min(window_stacks)
    max_stack = max(window_stacks)
    left_bonds: List[int] = []
    right_bonds: List[int] = []
    fixed_edge_bonds: List[int] = []
    for u in sorted_old_indices:
        for neighbor in original_mol.GetAtomWithIdx(u).GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in monomer_atoms:
                continue
            neighbor_stacks = atom_to_stacks.get(n_idx, set())
            if not neighbor_stacks:
                continue
            if max(neighbor_stacks) < min_stack:
                left_bonds.append(u)        # 周期方向（左）→ Br
            elif min(neighbor_stacks) > max_stack:
                right_bonds.append(u)       # 周期方向（右）→ Br
            else:
                fixed_edge_bonds.append(u)  # 固定边（上/下）→ 甲基/H

    if not left_bonds or not right_bonds:
        return []

    # 固定边断键：每个都封一个甲基（C）。armchair 竖直 fused 堆上下端通常为
    # 完整环边、无固定边断键（fixed_edge_bonds 为空）→ 0 甲基，合法。
    conf = original_mol.GetConformer()
    fixed_edge_bonds = sorted(fixed_edge_bonds, key=lambda u: conf.GetAtomPosition(u).x)

    capped_mols = []
    for left in left_bonds:
        for right in right_bonds:
            rw_mol = Chem.RWMol(base_rw_mol)
            for u in (left, right):
                cap = rw_mol.AddAtom(Chem.Atom("Br"))
                rw_mol.AddBond(old_to_new_map[u], cap, Chem.BondType.SINGLE)
            for u in fixed_edge_bonds:
                cap = rw_mol.AddAtom(Chem.Atom("C"))
                rw_mol.AddBond(old_to_new_map[u], cap, Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(rw_mol)
                capped_mols.append(rw_mol.GetMol())
            except Exception:
                pass
    return capped_mols


def _generate_armchair_monomers(
    original_mol,
    all_hexes: List[BenzeneHex],
    raw_smi_filename: str,
    capped_smi_filename: str,
    img_filename: str,
) -> MonomerGenerationResult:
    """armchair 专用周期单元生成：基于竖直堆窗口，周期方向封 Br。

    窗口宽度 = 周期堆数 + 1（实测：捕获完整一个周期 + 衔接 biaryl 单键所需）。
    与 zigzag 路径完全隔离，不共用窗口/封端逻辑。
    """
    result = MonomerGenerationResult()

    stack_of, num_stacks = _assign_vertical_stacks(all_hexes)
    if num_stacks == 0:
        result.failure_reason = "armchair: no vertical stacks detected"
        return result

    heights = _vertical_stack_heights(all_hexes)
    period_stacks = _minimal_sequence_period(heights)
    window_width = period_stacks + 1
    if window_width > num_stacks:
        result.failure_reason = "armchair: ribbon too short for one periodic window"
        return result

    atom_to_stacks: Dict[int, set] = {}
    for h in all_hexes:
        for aid in h.atom_indices:
            atom_to_stacks.setdefault(aid, set()).add(stack_of[h.id])

    raw_results = []
    unique_raw_smiles = set()
    capped_results = []
    unique_capped_smiles = set()

    for start_stack in range(num_stacks - window_width + 1):
        window_stacks = set(range(start_stack, start_stack + window_width))
        window_hexes = [h for h in all_hexes if stack_of[h.id] in window_stacks]
        if not window_hexes:
            continue

        raw_mol_rw = extract_submol_from_hexes(original_mol, window_hexes)
        if raw_mol_rw:
            frags = Chem.GetMolFrags(raw_mol_rw.GetMol(), asMols=True, sanitizeFrags=False)
            if frags:
                best_raw_mol = max(frags, key=lambda m: m.GetNumAtoms())
                try:
                    Chem.SanitizeMol(best_raw_mol)
                    smi = Chem.MolToSmiles(best_raw_mol)
                    if smi not in unique_raw_smiles:
                        unique_raw_smiles.add(smi)
                        raw_results.append(best_raw_mol)
                except Exception:
                    pass

        capped_mols = _extract_armchair_capped_monomers(
            original_mol, window_hexes, window_stacks, atom_to_stacks
        )
        for capped_mol in capped_mols:
            frags = Chem.GetMolFrags(capped_mol, asMols=True, sanitizeFrags=True)
            if not frags:
                continue
            best = max(frags, key=lambda m: m.GetNumAtoms())
            if best.GetNumAtoms() != capped_mol.GetNumAtoms():
                continue  # 必须是连通单分子
            if not _accept_capped_monomer(best, "armchair"):
                continue
            smi = Chem.MolToSmiles(best)
            if smi in unique_capped_smiles:
                continue
            unique_capped_smiles.add(smi)
            capped_results.append(best)

    if not raw_results:
        result.failure_reason = "armchair: no valid raw monomer smiles generated"
        return result
    if not capped_results:
        result.failure_reason = "armchair: no valid capped monomer smiles generated"
        return result

    _write_monomer_outputs(
        result, raw_results, capped_results,
        raw_smi_filename, capped_smi_filename, img_filename,
    )
    return result


def _write_monomer_outputs(
    result: MonomerGenerationResult,
    raw_results: List[Chem.Mol],
    capped_results: List[Chem.Mol],
    raw_smi_filename: str,
    capped_smi_filename: str,
    img_filename: str,
) -> None:
    """把 raw / capped 产物写盘（与 zigzag 末段写盘逻辑等价，供两分支共用）。"""
    for idx, mol in enumerate(raw_results):
        smi = Chem.MolToSmiles(mol)
        out_name = raw_smi_filename if len(raw_results) == 1 else raw_smi_filename.replace(".smi", f"_{idx+1}.smi")
        try:
            with open(out_name, 'w') as f:
                f.write(smi)
            result.raw_smiles.append(smi)
            result.raw_files.append(out_name)
            print(f"    [成功] 原始骨架保存: {os.path.basename(out_name)}")
        except Exception as exc:
            result.failure_reason = f"failed to write raw smiles: {exc}"
            return

    for idx, mol in enumerate(capped_results):
        smi = Chem.MolToSmiles(mol)
        if len(capped_results) == 1:
            out_smi_name = capped_smi_filename
            out_img_name = img_filename
        else:
            smi_p = Path(capped_smi_filename)
            out_smi_name = str(smi_p.parent / f"{smi_p.stem}_{idx+1}{smi_p.suffix}")
            img_p = Path(img_filename)
            out_img_name = str(img_p.parent / f"{img_p.stem}_{idx+1}{img_p.suffix}")
        try:
            with open(out_smi_name, 'w') as f:
                f.write(smi)
            AllChem.Compute2DCoords(mol)
            Draw.MolToFile(mol, out_img_name, size=(600, 600))
            result.capped_smiles.append(smi)
            result.capped_files.append(out_smi_name)
            result.monomer_images.append(out_img_name)
            print(f"    [成功] 智能封端保存: {os.path.basename(out_smi_name)} | 图像: {os.path.basename(out_img_name)}")
        except Exception as exc:
            result.failure_reason = f"failed to write capped smiles or image: {exc}"
            return

    result.is_valid = bool(result.raw_smiles and result.capped_smiles)
    if not result.is_valid:
        result.failure_reason = "monomer output files were incomplete"


def generate_monomer_smiles_periodic(original_mol, all_hexes: List[BenzeneHex],
                                     global_cutting_plan: Dict[int, List[Tuple[int, int]]] | GlobalCutPlan,
                                     k: int, max_col: int,
                                     raw_smi_filename: str, capped_smi_filename: str, img_filename: str,
                                     edge_type: str = "zigzag") -> MonomerGenerationResult:
    # armchair：竖直堆错开半周期，grid-col 窗口无法 kekulize，走专用周期单元路径。
    # zigzag（默认）：保持与原实现逐字节等价的代码路径。
    if edge_type == "armchair":
        return _generate_armchair_monomers(
            original_mol, all_hexes,
            raw_smi_filename, capped_smi_filename, img_filename,
        )

    result = MonomerGenerationResult()
    top_exit_direction = ""
    cutting_edges = global_cutting_plan
    if isinstance(global_cutting_plan, GlobalCutPlan):
        top_exit_direction = global_cutting_plan.top_exit_direction or ""
        cutting_edges = global_cutting_plan.cutting_edges

    cut_boundary_hex_ids, cut_atom_bonds = _collect_cut_boundary(original_mol, all_hexes, cutting_edges)
    if not cut_atom_bonds:
        result.failure_reason = "no molecular cut bonds mapped from cut path"
        return result

    window_candidates = []
    
    for start_col in range(max_col - k + 1):
        window_hexes = [
            h for h in all_hexes
            if h.id not in cut_boundary_hex_ids and start_col <= h.col < start_col + k
        ]
        if not window_hexes: continue
        if not _hexes_form_connected_fused_component(window_hexes): continue

        mol = extract_submol_from_hexes(original_mol, window_hexes)
        if not mol: continue

        frags = Chem.GetMolFrags(mol.GetMol(), asMols=True, sanitizeFrags=False)
        if not frags: continue
        
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        num_atoms = largest_frag.GetNumAtoms()
        window_candidates.append((start_col, window_hexes, num_atoms))

    if not window_candidates:
        result.failure_reason = "no valid monomer windows generated"
        return result

    interior_candidates = [
        (start_col, window_hexes, num_atoms)
        for start_col, window_hexes, num_atoms in window_candidates
        if start_col > 0 and start_col + k < max_col
    ]
    candidate_pool = interior_candidates
    if not candidate_pool:
        result.failure_reason = "no interior periodic monomer windows generated"
        return result
    max_atoms = max(num_atoms for _start_col, _window_hexes, num_atoms in candidate_pool)
    monomer_windows = [
        (start_col, window_hexes)
        for start_col, window_hexes, num_atoms in candidate_pool
        if num_atoms == max_atoms
    ]

    unique_raw_smiles = set()
    raw_results = []
    capped_results = []

    for _start_col, window_hexes in monomer_windows:
        raw_mol_rw = extract_submol_from_hexes(original_mol, window_hexes)
        if raw_mol_rw:
            frags = Chem.GetMolFrags(raw_mol_rw.GetMol(), asMols=True, sanitizeFrags=False)
            if frags:
                best_raw_mol = max(frags, key=lambda m: m.GetNumAtoms())
                try:
                    Chem.SanitizeMol(best_raw_mol)
                    smi = Chem.MolToSmiles(best_raw_mol)
                    if smi not in unique_raw_smiles:
                        unique_raw_smiles.add(smi)
                        raw_results.append(best_raw_mol)
                except: pass

    best_capped_count = 0
    best_capped_results = []
    for _start_col, target_window in monomer_windows:
        capped_mols = extract_all_capped_monomers(
            original_mol,
            target_window,
            all_hexes,
            cut_atom_bonds,
            top_exit_direction=top_exit_direction,
        )
        for capped_mol in capped_mols:
            frags = Chem.GetMolFrags(capped_mol, asMols=True, sanitizeFrags=True)
            if frags:
                best_capped_mol = max(frags, key=lambda m: m.GetNumAtoms())
                capped_results.append(best_capped_mol)
        if len(capped_results) > best_capped_count:
            best_capped_count = len(capped_results)
            best_capped_results = capped_results
        capped_results = []
    capped_results = best_capped_results

    if not raw_results:
        result.failure_reason = "no valid raw monomer smiles generated"
        return result

    if not capped_results:
        result.failure_reason = "no valid capped monomer smiles generated"
        return result

    capped_results = [
        mol for mol in capped_results
        if _accept_capped_monomer(mol, "zigzag")
    ]
    if not capped_results:
        result.failure_reason = "no dibrominated dimethyl capped monomer smiles generated"
        return result

    unique_capped_results = []
    unique_capped_smiles = set()
    for mol in capped_results:
        smi = Chem.MolToSmiles(mol)
        if smi in unique_capped_smiles:
            continue
        unique_capped_smiles.add(smi)
        unique_capped_results.append(mol)
    capped_results = unique_capped_results

    # 保存文件。先确认 capped 产物存在，再写 raw，避免留下 raw-only 半成品 artifact。
    for idx, mol in enumerate(raw_results):
        smi = Chem.MolToSmiles(mol)
        out_name = raw_smi_filename if len(raw_results) == 1 else raw_smi_filename.replace(".smi", f"_{idx+1}.smi")
        try:
            with open(out_name, 'w') as f:
                f.write(smi)
            result.raw_smiles.append(smi)
            result.raw_files.append(out_name)
            print(f"    [成功] 原始骨架保存: {os.path.basename(out_name)}")
        except Exception as exc:
            result.failure_reason = f"failed to write raw smiles: {exc}"
            return result

    for idx, mol in enumerate(capped_results):
        smi = Chem.MolToSmiles(mol)
        if len(capped_results) == 1:
            out_smi_name = capped_smi_filename
            out_img_name = img_filename
        else:
            smi_p = Path(capped_smi_filename)
            out_smi_name = str(smi_p.parent / f"{smi_p.stem}_{idx+1}{smi_p.suffix}")
            img_p = Path(img_filename)
            out_img_name = str(img_p.parent / f"{img_p.stem}_{idx+1}{img_p.suffix}")
        try:
            with open(out_smi_name, 'w') as f:
                f.write(smi)
            AllChem.Compute2DCoords(mol)
            Draw.MolToFile(mol, out_img_name, size=(600, 600))
            result.capped_smiles.append(smi)
            result.capped_files.append(out_smi_name)
            result.monomer_images.append(out_img_name)
            print(f"    [成功] 智能封端保存: {os.path.basename(out_smi_name)} | 图像: {os.path.basename(out_img_name)}")
        except Exception as exc:
            result.failure_reason = f"failed to write capped smiles or image: {exc}"
            return result

    result.is_valid = bool(result.raw_smiles and result.capped_smiles)
    if not result.is_valid:
        result.failure_reason = "monomer output files were incomplete"
    return result
