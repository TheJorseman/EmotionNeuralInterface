
def split_data_by_len(data, n):
    for i in range(0, len(data), n):  
        yield data[i:i + n] 
