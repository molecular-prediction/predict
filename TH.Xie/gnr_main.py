import os
import shutil
from gnr_graph import read_smiles_and_generate_coords, mol_to_hex_grid, build_edges_and_adj_geometric, partition_into_tiles, apply_path_to_all_tiles
from gnr_pathfinder import EdgeCuttingPathFinder
from gnr_smiles import generate_monomer_smiles_periodic
from gnr_visualizer import draw_multi_cut_result

def clean_pycache():
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                path = os.path.join(root, d)
                try:
                    shutil.rmtree(path)
                except Exception:
                    pass

def main():
    input_file = "smile/gnr_7ac_segment.smi"
    
    # 建立分类明确的输出文件夹
    photo_dir = "photo"                          # 切割方案网格图
    smile_raw_dir = "predict_smile_raw"          # [原] 不加任何封端的纯骨架 SMILES
    smile_capped_dir = "predict_smile_capped"    # [新] 自动添加 -CH3 和 -Br 的 SMILES
    monomer_img_dir = "monomer_imgs"             # 封端后的 RDKit 标准化学结构图
    
    os.makedirs(photo_dir, exist_ok=True)
    os.makedirs(smile_raw_dir, exist_ok=True)
    os.makedirs(smile_capped_dir, exist_ok=True)
    os.makedirs(monomer_img_dir, exist_ok=True)

    max_k_attempts = 5

    print(">>> 步骤1: 读取并构建全局图...")
    try:
        mol = read_smiles_and_generate_coords(input_file)
        hexes, total_width = mol_to_hex_grid(mol)
        edges, all_adj = build_edges_and_adj_geometric(hexes)
        print(f"    分子总宽: {total_width} 列, 苯环数: {len(hexes)}")
    except Exception as e:
        print(f"出错: {e}")
        clean_pycache()
        exit()

    print(f">>> 步骤2: 开始尝试多种切割方案 (K=1 ~ {max_k_attempts})...")
    found_any_global = False

    for k in range(1, max_k_attempts + 1):
        if k > total_width: continue

        tiles = partition_into_tiles(hexes, k_cols=k)
        if not tiles or not tiles.get(0): continue
        template_tile = tiles.get(0)

        if len({h.row for h in template_tile}) <= 1: continue

        template_finder = EdgeCuttingPathFinder(template_tile, all_adj, k_cols=k)
        all_possible_paths = template_finder.find_all_paths()

        if not all_possible_paths: continue

        print(f"    K={k}: 模板找到 {len(all_possible_paths)} 种路径，正在处理...")

        variant_count = 0
        for path in all_possible_paths:
            global_plan = apply_path_to_all_tiles(path, template_tile, tiles, all_adj)

            if global_plan:
                variant_count += 1
                found_any_global = True

                base_name = f"cut_method_k{k}_v{variant_count}"
                
                img_path = os.path.join(photo_dir, base_name + ".png")
                raw_smi_path = os.path.join(smile_raw_dir, base_name + "_raw.smi")
                capped_smi_path = os.path.join(smile_capped_dir, base_name + "_capped.smi")
                monomer_img_path = os.path.join(monomer_img_dir, base_name + "_monomer.png")

                draw_multi_cut_result(hexes, global_plan, k, variant_count, img_path)

                # 将三个文件路径都传给处理函数
                generate_monomer_smiles_periodic(
                    mol, hexes, global_plan, k, total_width,
                    raw_smi_path, capped_smi_path, monomer_img_path
                )

                if variant_count >= 5: break

    if not found_any_global:
        print("\n未找到任何有效的切割方案。")
    else:
        print(f"\n全部完成！请检查输出文件夹。")
    
    clean_pycache()

if __name__ == "__main__":
    main()
