def divide_arr(arr, start, end):
    if end-start<=1:
        return None
    elif end-start==2:
        s=start
        e = end-1
        if arr[s]>arr[e]:
            tmp=arr[s]
            arr[s]=arr[e]
            arr[e] = tmp
        return None

    pivot = arr[start]

    s = start-1
    e=end

    while True:
        s+=1
        if arr[s]<pivot:
            continue

        while True:
            e-=1
            if e<=s:
                return s
            if arr[e]>pivot:
                continue

            tmp = arr[s]
            arr[s] = arr[e]
            arr[e] = tmp
            break


def sort_start_end(arr, start, end):
    print '{}->{}'.format(start, end)
    pos = divide_arr(arr, start, end)
    if pos is None:
        return

    sort_start_end(arr, start, pos)
    sort_start_end(arr, pos, end)

def sort_arr(arr):
    sort_start_end(arr, 0, len(arr))

