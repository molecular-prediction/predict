import logging
import os

from gnr_service import clean_pycache, run_pipeline


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

def main():
    input_file = "smile/gnr_7ac_segment.smi"
    result = run_pipeline(input_file, max_k_attempts=999)
    if not result.found_any_global:
        print("\n未找到任何有效的切割方案。")
    else:
        print("\n全部完成！请检查输出文件夹。")
    clean_pycache()

if __name__ == "__main__":
    main()
