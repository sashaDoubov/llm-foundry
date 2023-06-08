import os
import pandas as pd
from composer.utils import get_file, parse_uri, maybe_create_object_store_from_uri


def main():
    local_relative_path = "~/overruling.csv"
    local_path = os.path.expanduser(local_relative_path)
    remote_path = "oci://mosaicml-internal-datasets/pile-of-law/overruling.csv"
    
    local_jsonl_big_relative_path = "~/processed_overruling.jsonl"
    local_jsonl_big_path = os.path.expanduser(local_jsonl_big_relative_path)
    upload_path_big = "oci://mosaicml-internal-datasets/pile-of-law/processed_overruling.jsonl"

    local_jsonl_small_relative_path = "~/processed_overruling_small.jsonl"
    local_jsonl_small_path = os.path.expanduser(local_jsonl_small_relative_path)
    upload_path_small = "oci://mosaicml-internal-datasets/pile-of-law/processed_overruling_small.jsonl"

    # local_jsonl_1k_relative_path = "~/processed_casehold_small_1k.jsonl"
    # local_jsonl_1k_path = os.path.expanduser(local_jsonl_1k_relative_path)
    # upload_path_1k = "oci://mosaicml-internal-datasets/pile-of-law/processed_casehold_1k.jsonl"


    if not os.path.exists(local_path):
        # print(local_path)
        get_file(remote_path, local_path) 
    df_full = pd.read_csv(local_path)

    df_full = df_full.assign(yes='yes', no='no')
    print(df_full.head())


    for df, local_jsonl_path, upload_path in ([df_full.head(), local_jsonl_small_path, upload_path_small],[df_full, local_jsonl_big_path, upload_path_big]):
        df_filtered = df.dropna()

        # need to be careful with order, here: yes -> 1 (is overruling)
        json_list = df_filtered.apply(lambda row: {'query': row['sentence1'], 'choices': [row['no'], row['yes'],], 'gold': int(row['label'])}, axis=1).tolist()
        json_list[0]
        print(len(json_list))
        json_df = pd.DataFrame(json_list)
        print(json_df.head())
    #     # Save the DataFrame as a JSON Lines file (.jsonl)
    #     # json_df.to_json(local_jsonl_path, orient='records', lines=True)
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