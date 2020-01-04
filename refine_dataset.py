# %%
# import pretty_errors
import xml.etree.ElementTree as ET
import os
import json

root_path = "/mnt/e/robomaster/DJI ROCO"  # change to your own path
dump_path = root_path
regions = [os.path.join(root_path, x) for x in os.listdir(
    root_path) if os.path.isdir(os.path.join(root_path, x))]
# %%


def test():
    for region in regions:
        os.chdir(region)
        print("working on %s" % region)

        # image_path = os.path.join(region, "image")
        annotation_path = os.path.join(region, "image_annotation")

        # for each annotation file, find the arnor information
        # and dump as json file
        # also cut the useful part of image out
        for file in os.listdir(annotation_path):
            print("refining annotation: "+file)
            xml_tree = ET.parse(os.path.join(annotation_path, file))
            root = xml_tree.getroot()
            for object_ in root.findall('object'):
                print("find an object")

                if object_.find('name').text == "armor":
                    for child in object_:
                        print(child)
                        # collect useful information
                        # dump to json
                        # cut image (corespondedly)
                else:
                    print('not armor object')


            break
        break
    return


if __name__ == "__main__":
    # main()
    test()


# %%
