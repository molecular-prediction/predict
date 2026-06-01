import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from gnr_types import BenzeneHex, GlobalCutPlan


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
    candidate_sites = []
    for old_idx in sorted_old_indices:
        atom = original_mol.GetAtomWithIdx(old_idx)
        if atom.GetSymbol() != "C" or not atom.GetIsAromatic():
            continue
        # A degree-2 aromatic carbon is a peripheral C-H site in the source PAH.
        # These are valid aryl bromide precursor positions; degree-3 fused atoms are not.
        if atom.GetDegree() == 2:
            pos = conf.GetAtomPosition(old_idx)
            candidate_sites.append((old_idx, pos.x, pos.y))

    if len(candidate_sites) < 2:
        return []

    candidate_sites.sort(key=lambda item: (item[1], item[2]))
    left_sites = [item for item in candidate_sites if item[1] <= candidate_sites[len(candidate_sites) // 2][1]]
    right_sites = [item for item in candidate_sites if item[1] > candidate_sites[len(candidate_sites) // 2][1]]

    br_pairs = []
    if left_sites and right_sites:
        for left in left_sites:
            for right in right_sites:
                br_pairs.append((left[0], right[0]))
    else:
        for i, first in enumerate(candidate_sites):
            for second in candidate_sites[i + 1:]:
                br_pairs.append((first[0], second[0]))

    def pair_score(pair):
        p1 = conf.GetAtomPosition(pair[0])
        p2 = conf.GetAtomPosition(pair[1])
        return (-abs(p1.x - p2.x), abs(p1.y - p2.y), p1.x + p2.x)

    br_pairs = sorted(set(tuple(sorted(pair)) for pair in br_pairs), key=pair_score)[:12]

    capped_mols = []
    for br_pair in br_pairs:
        rw_mol = Chem.RWMol(base_rw_mol)
        for atom_idx in br_pair:
            cap = rw_mol.AddAtom(Chem.Atom("Br"))
            rw_mol.AddBond(old_to_new_map[atom_idx], cap, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw_mol)
            capped_mols.append(rw_mol.GetMol())
        except Exception:
            pass

    return capped_mols

def generate_monomer_smiles_periodic(original_mol, all_hexes: List[BenzeneHex],
                                     global_cutting_plan: Dict[int, List[Tuple[int, int]]] | GlobalCutPlan,
                                     k: int, max_col: int,
                                     raw_smi_filename: str, capped_smi_filename: str, img_filename: str) -> MonomerGenerationResult:
    result = MonomerGenerationResult()
    top_exit_direction = ""
    cutting_edges = global_cutting_plan
    if isinstance(global_cutting_plan, GlobalCutPlan):
        top_exit_direction = global_cutting_plan.top_exit_direction or ""
        cutting_edges = global_cutting_plan.cutting_edges

    _cut_boundary_hex_ids, cut_atom_bonds = _collect_cut_boundary(original_mol, all_hexes, cutting_edges)
    if not cut_atom_bonds:
        result.failure_reason = "no molecular cut bonds mapped from cut path"
        return result

    window_candidates = []
    
    for start_col in range(0, max_col - k + 1, k):
        window_hexes = [
            h for h in all_hexes
            if start_col <= h.col < start_col + k
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
    candidate_pool = interior_candidates or window_candidates
    if not candidate_pool:
        result.failure_reason = "no periodic monomer windows generated"
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

    raw_aromatic_ring_count = max(_count_aromatic_rings(mol) for mol in raw_results)
    capped_results = [
        mol for mol in capped_results
        if _count_aromatic_rings(mol) >= raw_aromatic_ring_count
    ]

    unique_capped_results = []
    unique_capped_smiles = set()
    for mol in capped_results:
        smi = Chem.MolToSmiles(mol)
        if smi in unique_capped_smiles:
            continue
        unique_capped_smiles.add(smi)
        unique_capped_results.append(mol)
    capped_results = unique_capped_results

    # 保存文件
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

    if not capped_results:
        for idx, mol in enumerate(raw_results):
            out_img_name = img_filename if len(raw_results) == 1 else img_filename.replace(".png", f"_{idx+1}.png")
            try:
                AllChem.Compute2DCoords(mol)
                Draw.MolToFile(mol, out_img_name, size=(600, 600))
                result.monomer_images.append(out_img_name)
            except Exception:
                pass
        result.is_valid = bool(result.raw_smiles)
        result.failure_reason = "no valid capped monomer smiles generated; wrote raw monomer image"
        return result

    for idx, mol in enumerate(capped_results):
        smi = Chem.MolToSmiles(mol)
        out_smi_name = capped_smi_filename if len(capped_results) == 1 else capped_smi_filename.replace(".smi", f"_{idx+1}.smi")
        out_img_name = img_filename if len(capped_results) == 1 else img_filename.replace(".png", f"_{idx+1}.png")
        
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
