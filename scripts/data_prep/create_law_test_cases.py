import os
import pandas as pd
from composer.utils import get_file, parse_uri, maybe_create_object_store_from_uri


def main():
    local_relative_path = "~/casehold.csv"
    local_path = os.path.expanduser(local_relative_path)
    remote_path = "oci://mosaicml-internal-datasets/pile-of-law/casehold.csv"
    
    local_jsonl_big_relative_path = "~/processed_casehold.jsonl"
    local_jsonl_big_path = os.path.expanduser(local_jsonl_big_relative_path)
    upload_path_big = "oci://mosaicml-internal-datasets/pile-of-law/processed_casehold.jsonl"

    local_jsonl_small_relative_path = "~/processed_casehold_small.jsonl"
    local_jsonl_small_path = os.path.expanduser(local_jsonl_small_relative_path)
    upload_path_small = "oci://mosaicml-internal-datasets/pile-of-law/processed_casehold_small.jsonl"

    local_jsonl_1k_relative_path = "~/processed_casehold_small_1k.jsonl"
    local_jsonl_1k_path = os.path.expanduser(local_jsonl_1k_relative_path)
    upload_path_1k = "oci://mosaicml-internal-datasets/pile-of-law/processed_casehold_1k.jsonl"


    if not os.path.exists(local_path):
        # print(local_path)
        get_file(remote_path, local_path) 
    df_full = pd.read_csv(local_path)


    for df, upload_path, local_jsonl_path in ([df_full.head(), upload_path_small,local_jsonl_small_path], [df_full, upload_path_big, local_jsonl_big_path], [df_full.sample(1).head(1000), upload_path_1k, local_jsonl_1k_path]):
        df_filtered = df.dropna()

        json_list = df_filtered.apply(lambda row: {'query': row['1'], 'choices': [row['2'], row['3'], row['4'], row['5'], row['6'] ], 'gold': int(row['12'])}, axis=1).tolist()
        json_list[0]
        print(len(json_list))
        json_df = pd.DataFrame(json_list)
        # Save the DataFrame as a JSON Lines file (.jsonl)
        # json_df.to_json(local_jsonl_path, orient='records', lines=True)
        with open(local_jsonl_path, 'w', encoding='utf-8') as f:
            for item in json_list:
                json_string = pd.io.json.dumps(item, ensure_ascii=False)
                f.write(json_string)
                f.write('\n')

        object_store = maybe_create_object_store_from_uri(upload_path)

        if object_store is not None:
            # remove bucket name and upper part of oci path
            _, _, prefix = parse_uri(upload_path)
            print(prefix)
            object_store.upload_object(prefix, local_jsonl_path)

    

if __name__ == "__main__":
    main()