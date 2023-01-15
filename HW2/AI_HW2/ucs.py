import csv
import heapq
edgeFile = 'edges.csv'

'''
I use the priority queue function in heapq library. Here is the class that i
put in the pq containing the id of the node and the cost to get to it, and also
redine the < operator for the Node class.
'''
class Node(object):
    def __init__(self, Id:int,val: float):
        self.val = val
        self.Id = Id

    def __repr__(self):
        return f'Node id: {self.Id} Node value: {self.val}'

    def __lt__(self, other):
        return self.val < other.val


def ucs(start, end):
    # Begin your code (Part 3)
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
    This is ucs part. Keep popping out the node that has the smallest cost to get to until
    we reach the end node or the pq is empty. Here we don't have to calculate the distance 
    of the path, because it will just be equivalent to the end node's cost when we pop it
    from pq.  
    '''
    n=0
    pq=[]
    heapq.heappush(pq,Node(start,0))
    seen=set()
    parent={Node(start,0):None}
    while len(pq):
        node=heapq.heappop(pq)
        if node.Id in seen:
            continue
        n+=1
        seen.add(node.Id)
        if node.Id==end:
            dist=node.val
            v=node
            break
        nei=graph.get(node.Id,-1)
        if nei==-1:
            continue
        for x in nei:
            if not (x[0] in seen) :
                temp=Node(x[0], x[1]+node.val)
                heapq.heappush(pq,temp)
                parent[temp]=node
    path=[]
    while v!=None:
        path.insert(0,v.Id)
        v=parent.get(v)
    num_visited=n
    return path, dist, num_visited
    # raise NotImplementedError("To be implemented")
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
