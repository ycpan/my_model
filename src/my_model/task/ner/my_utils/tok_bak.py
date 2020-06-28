def tok(line,tag='O'):
    SEP='!,.:;?'
    li=[]
    s=[]
    line = line.strip()
    for c in list(line):
        if c in ' ':
            if li:
                s.append(li)
                li = []
            continue
        if any([c==i for i in SEP]):
           if li:
                s.append(li)
                li = []
           s.append(['{}'.format(c)])
        else:
            li.append(c)
    #for comma,special process
    res = []
    length= len(s)
    i = 0
    while i < length - 1:
        c = ''.join(s[i])
        if c in SEP:
            c_1 = ''.join(s[i+1])
            if c_1 == '/O':  
                res.append(''.join([c,c_1]))
                i = i + 2
                continue
            else:
                res.append(c + '/O')
        else:
            res.append(c)
        i += 1
    while i < length:
        c = ''.join(s[i])
        if c in SEP:
            res.append(c + '/O')
        else:
            res.append(c)
        i = i+1
    return res
            
