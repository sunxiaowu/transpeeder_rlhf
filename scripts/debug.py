import os

strs = ["eat","tea","tan","ate","nat","bat"]
strs = [""]
strs = ["a"]
res_list = list()
for item in strs:
    item_map = dict()
    for _item in item:
        if _item not in item_map:
            item_map[_item] = 1
        else:
            item_map[_item] += 1
    item_str = "".join(sorted(item_map.keys()))
    if len(res_list) == 0:
        res_list.append([item_str, item])
    else:
        flag = False
        for _res in res_list:
            if _res[0] == item_str:
                _res.append(item)
                flag = True
        if flag is False:
            res_list.append([item_str, item])
res_list = [_[1:] for _ in res_list]
print(res_list)