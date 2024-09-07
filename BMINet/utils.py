def convert_to_number(lst):
    result = []
    for item in lst:
        if item == '':
            result.append(None)
        else:
            try:
                result.append(float(item))
            except ValueError:
                result.append(item)
    return result