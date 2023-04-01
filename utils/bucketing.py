def up_down_bucket(m_size, in_size, direction):
    if direction == 'down': return abs(int(m_size - in_size))
    if direction == 'up': return abs(int(m_size + in_size))

def get_bucket_sizes(size, direction: 'down'):
    multipliers = [0, 64, 128]
    for i, m in enumerate(multipliers):
        res =  up_down_bucket(m, size, direction)
        multipliers[i] = res
    
    return multipliers

def closest_bucket(m_size, size, direction):
    lst = get_bucket_sizes(m_size, direction)
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i]-size))]

def resolve_bucket(i,h,w): return  (i / (h / w))

def sensible_buckets(m_width, m_height, w, h):
    if h > w:
        w = resolve_bucket(m_width, h, w)
        w = closest_bucket(m_width, w, 'down')
        return w, m_height
    if h < w:
        h = resolve_bucket(m_height, w, h)
        h = closest_bucket(m_height, h, 'down')
        return m_width, h

    return m_width, m_height