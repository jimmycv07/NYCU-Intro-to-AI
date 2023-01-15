import csv
import heapq
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

'''
Class in astar is modified to contain 3 variables, which are id, cost(dist) and
cost+heuristic(dist from start to node + euclidean dist from node to end)
'''
class Node(object):
    def __init__(self, Id:int,d:float,val: float):
        self.val = val
        self.Id = Id
        self.d = d

    def __repr__(self):
        return f'Node id: {self.Id} Node value: {self.val}'

    def __lt__(self, other):
        return self.val < other.val


def astar(start, end):
    # Begin your code (Part 4)
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
    This part is to input the Euclidean distance data, which will be use in 
    both astar(distance) and astar(time) version. I store the data in a dictionary
    of another dictionary. The first key is the id of the end node, and the second
    key is the node id where I want to start from. By this structure, I dont have to
    change anything except for the input arguments of the function. 
    '''
    heu={}
    fptr=open(heuristicFile,'r')
    nID=fptr.readline().split(',')
    for i in range(1,4):
        heu[int(nID[i])]={}
   
    while 1:
        temp=fptr.readline()
        if not temp:
            break
        temp=temp.split(',')
        for i in range(1,4):
            heu[int(nID[i])][int(temp[0])]=float(temp[i])
    fptr.close()
    '''
    This is the astar(distance) part, which is exactly the same as the ucs one, except
    that we compare two Nodes by dist from start to node + euclidean dist from node to end.
    '''
    n=0
    pq=[]
    heapq.heappush(pq,Node(start,0,heu[end][start]))
    seen=set()
    parent={Node(start,0,heu[end][start]):None}
    while len(pq):
        node=heapq.heappop(pq)
        # print(node)
        if node.Id in seen:
            continue
        n+=1
        if node.Id==end:
            dist=node.d
            v=node
            break
        seen.add(node.Id)
        nei=graph.get(node.Id,-1)
        if nei==-1 :
            continue
        for x in nei:
            if not(x[0] in seen):
                temp=Node(x[0],node.d+x[1],node.d+x[1]+heu[end][x[0]]) 
                heapq.heappush(pq,temp)
                parent[temp]=node

    path=[]
    while v!=None:
        path.insert(0,v.Id)
        v=parent.get(v)
    num_visited=n
    return path, dist, num_visited
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
