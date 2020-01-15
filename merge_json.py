#%%
import json
import os

main_root_path = '/mnt/e/robomaster/mydump/'
regions = [x for x in os.listdir(main_root_path) if (
    os.path.isdir(os.path.join(main_root_path, x))and x != 'lightbar')]
for region in regions:
    root_path = os.path.join(main_root_path,region,'image_annotation')
    
    json_files = os.listdir(root_path)
    new_json ={}
    try:
        for json_file in json_files:
            with open(os.path.join(root_path, json_file)) as f:
                data = json.load(f)
            new_json.update(data)
    except:
        pass
    finally:
        with open(os.path.join(root_path,'..','merged_armor.json'),'w') as f:
            json.dump(new_json,f,ensure_ascii=False)

# %%
