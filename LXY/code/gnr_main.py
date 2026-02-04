import os
from gnr_graph import read_smiles_and_generate_coords, mol_to_hex_grid, build_edges_and_adj_geometric, partition_into_tiles, apply_path_to_all_tiles
from gnr_pathfinder import EdgeCuttingPathFinder
from gnr_smiles import generate_monomer_smiles_periodic
from gnr_visualizer import draw_multi_cut_result

def main():
    input_file = "smile/gnr_7ac_segment.smi"
    photo_dir = "photo"
    smile_dir = "predict_smile"
    os.makedirs(photo_dir, exist_ok=True)
    os.makedirs(smile_dir, exist_ok=True)

    max_k_attempts = 5

    print(">>> 步骤1: 读取并构建全局图...")
    try:
        mol = read_smiles_and_generate_coords(input_file)
        hexes, total_width = mol_to_hex_grid(mol)
        edges, all_adj = build_edges_and_adj_geometric(hexes)
        print(f"    分子总宽: {total_width} 列, 苯环数: {len(hexes)}")
    except Exception as e:
        print(f"出错: {e}")
        exit()

    print(f">>> 步骤2: 开始尝试多种切割方案 (K=1 ~ {max_k_attempts})...")
    found_any_global = False

    for k in range(1, max_k_attempts + 1):
        if k > total_width: continue

        tiles = partition_into_tiles(hexes, k_cols=k)
        if not tiles or not tiles.get(0): continue
        template_tile = tiles.get(0)

        if len({h.row for h in template_tile}) <= 1: continue

        template_finder = EdgeCuttingPathFinder(template_tile, all_adj)
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
                draw_multi_cut_result(hexes, global_plan, k, variant_count, img_path)

                # --- 核心修改 ---
                smi_path = os.path.join(smile_dir, base_name + ".smi")
                # 传入 total_width 用于定位中间区域
                generate_monomer_smiles_periodic(mol, hexes, global_plan, k, total_width, smi_path)

                if variant_count >= 5: break

    if not found_any_global:
        print("\n未找到任何有效的切割方案。")
    else:
        print(f"\n全部完成！\n图片路径: {photo_dir}\nSMILES路径: {smile_dir}")

if __name__ == "__main__":
    main()
