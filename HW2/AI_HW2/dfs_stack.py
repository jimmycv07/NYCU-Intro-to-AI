import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    fptr=open(edgeFile,'r')
    fptr.readline()
    graph={}
    neighbors=[]
    while 1:
        temp=fptr.readline()
        if not temp:
            break
        temp=temp.split(',')
        key=int(temp[0])
        if key not in graph.keys():
            neighbors=[]
        neighbors.append([int(temp[1]), float(temp[2])])
        graph[key]=neighbors
    fptr.close()
    '''
    This is dfs part. I do the iterative version by implementing the 
    stack structure property. It's worth noticed that browsing the neighbor
    nodes with backward order might more fit the way how we demonstrate dfs
    on the real graph. The rest parts are just exaclty same as bfs.
    '''
    n=0
    stack=[]
    stack.append(start)
    seen=set()
    parent={start:None}
    while len(stack):
        node=stack.pop()
        if (node in seen):
            continue
        n+=1
        seen.add(node)
        nei=graph.get(node,-1)
        if nei==-1:
            continue
        for x in nei[::-1]:
            if x[0]==end:
                n+=1
                parent[end]=node
                break
            elif not(x[0] in seen):
                stack.append(x[0])
                parent[x[0]]=node
        if (end in parent):
            break

    path=[]
    v=end
    dist=0
    while 1:
        path.insert(0,v)
        u=parent[v]
        if u==None:
            break
        for x in graph.get(u):
            if x[0]==v:
                dist+=x[1]
                break
        v=u
    num_visited=n
    return path, dist , num_visited
    # raise NotImplementedError("To be implemented")
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
