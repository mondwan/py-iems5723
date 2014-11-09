

lst = [
    ("news", "apple"),
    ("tech", "apple"),
    ("news", "gun"),
    ("tech", "acer")
]

def extFeature (item):
    print 'item', item
    return {"head": item[0]}

extracted = [(extFeature(val), key) for key, val in lst]

print 'extraced', extracted
