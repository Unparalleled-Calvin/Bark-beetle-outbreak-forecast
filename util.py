import numpy as np

def genResultByYear(dataset, year, M=7):

    dataset = dataset[(dataset[:,3] >= year - 3) * (dataset[:,3] <= year)]

    idlistlen = (dataset[:,3]==year).sum()
    
    matrixsize = np.ptp(dataset,axis=0)
    underborder = np.amin(dataset,axis=0)

    matrixsize[1] = matrixsize[1]/30
    matrixsize[2] = matrixsize[2]/30

    dataset[:,1] = (dataset[:,1] - underborder[1])/30
    dataset[:,2] = (dataset[:,2] - underborder[2])/30

    preprocessdata = np.empty([matrixsize[1]+1,matrixsize[2]+1],dtype = int)
    preprocessdata[:,:] = -1

    # initialize

    for record in dataset:
        if record[4] == 1:
            if (record[3] == year-1)and(preprocessdata[record[1],record[2]] < 3):
                preprocessdata[record[1],record[2]] = 3
            elif (record[3] == year-2)and(preprocessdata[record[1],record[2]] < 2):
                preprocessdata[record[1],record[2]] = 2
            elif (record[3] == year-3)and(preprocessdata[record[1],record[2]] < 1):
                preprocessdata[record[1],record[2]] = 1
            elif (preprocessdata[record[1],record[2]] == -1):
                preprocessdata[record[1],record[2]] = 0
        elif (record[4] == 0)and(preprocessdata[record[1],record[2]] == -1):
            preprocessdata[record[1],record[2]] = 0
    #preprocess,to get the tag of every point

    result = np.empty([idlistlen,M*M+1],dtype = int)

    for index in range(0,idlistlen):
        result[index,0] = dataset[index,0]
        pointer = 1
        for neighborx in range(-(M//2),M//2+1):
            for neighbory in range(-(M//2),M//2+1):
                axisx = dataset[index,1] + neighborx
                axisy = dataset[index,2] + neighbory
                if (axisx <= matrixsize[1])and(axisx >= 0)and(axisy <= matrixsize[2])and(axisy >= 0):
                    result[index,pointer] = preprocessdata[axisx,axisy]
                else:
                    result[index,pointer] = -1
                pointer = pointer + 1
      
    return result


def genResultByYears(years, M=7):
    if type(years) == int:
        years = [years]
    result = np.array([])
    dataset = np.loadtxt('./data.txt',dtype=int,skiprows = 1,delimiter=' ',usecols=(0,1,2,3,4))
    results = [genResultByYear(dataset, year, M) for year in years]
    results = np.concatenate(results, axis = 0)
    return results

if __name__ == "__main__":
    np.savetxt("result.csv", genResultByYears(2000), delimiter=",", fmt="%d")


