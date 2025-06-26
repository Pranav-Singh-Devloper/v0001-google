from bson import ObjectId

def oid_to_str(obj):
    """
    Recursively replace any bson.ObjectId in obj with its string form.
    Works on dicts, lists, and scalars.
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, list):
        return [oid_to_str(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: oid_to_str(v) for k, v in obj.items()}
    else:
        return obj
