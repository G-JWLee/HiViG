import pandas as pd

# Chosen
# 1. Load the dataframes
df_chosen = pd.read_json("/nas-ssd2/jwoolee/code/verl_vl/benchmarks/new_scalecua_critic_fixed/critic_chosen_dataset_fixed_grounding_merged_all_new.jsonl", lines=True)
df_chosen_no_verbal = pd.read_json("/nas-ssd2/jwoolee/code/verl_vl/benchmarks/new_scalecua_critic_fixed/critic_chosen_dataset_fixed_grounding_no_verbal.jsonl", lines=True)

# 2. Define the unique column that exists in both dataframes
id_column = 'image_path'

# 3. Set this unique column as the index for both dataframes
df_chosen.set_index(id_column, inplace=True)
df_chosen_no_verbal.set_index(id_column, inplace=True)

# 4. Update test with test2's data
df_chosen.update(df_chosen_no_verbal)

# 5. Reset the index to put the ID column back where it belongs
df_chosen.reset_index(inplace=True)



# Rejected
# 1. Load the dataframes
df_rejected = pd.read_json("/nas-ssd2/jwoolee/code/verl_vl/benchmarks/new_scalecua_critic_fixed/critic_rejected_dataset_fixed_grounding_merged_all_new.jsonl", lines=True)
df_rejected_no_verbal = pd.read_json("/nas-ssd2/jwoolee/code/verl_vl/benchmarks/new_scalecua_critic_fixed/critic_rejected_dataset_fixed_grounding_merged_all_new_no_verbal.jsonl", lines=True)

# 2. Define the unique column that exists in both dataframes
id_column = 'rejected_thought'

# 3. Set this unique column as the index for both dataframes
df_rejected.set_index(id_column, inplace=True)
df_rejected_no_verbal.set_index(id_column, inplace=True)

# 4. Update test with test2's data
df_rejected.update(df_rejected_no_verbal)

# 5. Reset the index to put the ID column back where it belongs
df_rejected.reset_index(inplace=True)


df_rejected.to_json("./benchmarks/new_scalecua_critic_fixed/critic_rejected_dataset.jsonl", orient="records", lines=True)
df_chosen.to_json("./benchmarks/new_scalecua_critic_fixed/critic_chosen_dataset.jsonl", orient="records", lines=True)

print("Done!")