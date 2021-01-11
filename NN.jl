using Random


mutable struct  Net 
    weights::Array{Float64}
    biases::Array{Float64}
    inOutDims::Array{Int64}
end

### Initializing a standard multilayer perceptron style network
### Rest of the functionality shouldn't be limited to this structure

function layerOf(x,dims)
    i = x
    l = 0
    for d in dims
        if i <= d
            return l
        end
        l += 1
        i -= d
    end
    println("Trying to handle node outwith network dimensions")
    return -1
end

function adjacentLayers(i, j, dims)
    l1 = layerOf(i,dims)
    l2 = layerOf(j,dims)
    return abs(l1-l2) == 1
end

function initWeights(dims)
    num = sum(dims)
    weights = zeros(Float64,num,num)
    for row in 1:num
        for col in 1:row
            if adjacentLayers(col, row, dims)
                weights[row,col] = (rand()*2) - 1.0
            end
        end
    end
    return weights
    #return [1.0 for col in 1:num if (adjacentLayers(col,row,dims) && col < row)] for row in 1:num]
end

function initBiases(dims)
    return [(rand()*2) - 1.0 for i in 1:(sum(dims))]
end


### initializing randomized network structure

function initWeightsRandom(inOutDims, numHiddenNodes, weightDensity)
    num = sum(inOutDims) + numHiddenNodes
    weights = zeros(Float64, num, num)
    for row in (inOutDims[1]+1):num
        for col in 1:(row-1)
            if rand() < weightDensity && col < (num-inOutDims[2])
                weights[row,col] = (rand()*2) - 1.0
            end
        end
    end
    return weights
end

# isn't working yet -----------------------
function initWeightsMultipleConvs(inOutDims, numConvLayers, numFullLayers)
    ### assume input is a flattened square image
    imageDim = sqrt(inOutDims[1])
    kernalSize = 3 # (3 x 3)
    stride = 2

    #convLayerDims = [(inOutDims[1]) ÷ (stride^(2*n)) for n in 1:numConvLayers]
    convLayerDims = [((imageDim+1)^2) ÷ (stride^(2*n)) for n in 1:numConvLayers]
    num = sum(inOutDims) + sum(convLayerDims) + inOutDims[2]*numFullLayers
    weights = zeros(Float64, Int32(num), Int32(num))

    

    startIndex = inOutDims[1] + 1
    prevDim = imageDim
    for l in 1:numConvLayers
        currDim = convLayerDims[l] #(prevDim+1) ÷ stride
        endIndex = startIndex + currDim^2
        for i in startIndex:endIndex
            indInPrevLayer = startIndex + ((i - startIndex)*stride + 1) - prevDim^2
            #layerCoord = (i ÷ currDim, i % currDim)
            println(indInPrevLayer)
            prevLayerCoord = (((indInPrevLayer-1) ÷ prevDim) + 1, ((indInPrevLayer-1) % prevDim) + 1)
            println(prevLayerCoord)
            for coord in getSurroundingCoords(prevLayerCoord, kernalSize, prevDim)
                println(coord)
                flatPrevLayerIndex = (coord[1] - 1)*prevDim + coord[2]
                weights[Int32(i), Int32(flatPrevLayerIndex)] = (rand()*2.0) - 1.0
            end
        end
        println("yo")
        startIndex = endIndex + 1
        println(startIndex)
        prevDim = currDim
        println(prevDim)
    end

    currDim = inOutDims[2]
    for l in 1:numFullLayers
        for i in 1:currDim
            to = startIndex + i
            for n in 1:prevDim
                
                from = startIndex - prevDim + n
                weights[to, from] = (rand()*2.0) - 1.0
            end
        end
        startIndex += currDim
        prevDim = currDim
    end

end

function getSurroundingCoords(coord, kernalSize, imageDim)
    leftRightAmount = kernalSize ÷ 2
    surroundingCoords = [coord]
    for x in (coord[1] - leftRightAmount):(coord[1] + leftRightAmount)
        for y in (coord[2] - leftRightAmount):(coord[2] + leftRightAmount)
            if validCoord((x,y), imageDim)
                append!(surroundingCoords, [(x,y)])
            end
        end
    end
    return surroundingCoords
end
function validCoord(coord, imageDim)
    return !(coord[1] < 1 || coord[1] > imageDim || coord[2] < 1 || coord[2] > imageDim)
end
# ------------------------------------------


### print the weight adjacency matrix
function printAdjacencyMatrix(net)
    num = size(net.weights,1)
    for n in 1:num
        if n <= net.inOutDims[1]
            outString = "  " * string(n) * " (in)  " * "\t|"
        elseif ((num - n) < net.inOutDims[2])
            outString = "  " * string(n) * " (out) " * "\t|"
        else 
            outString = "  " * string(n) * "       \t|"
        end
        for val in net.weights[n,:]
            if val != 0
                outString = outString * " x"
            else 
                outString = outString * " -"
            end
        end
        println(outString)
    end
end




# if we want a different activation function, for now, just edit these functions:
function activation(z)
    return tanh(z)
end
function activationPrime(z) # derivative of the activation function
    return tanhPrime(z)
end

### some activation functions and their derivatives
function sigmoid(z)
    return 1.0 / (1.0 + 2.718^(-z))
end

function sigmoidPrime(z)
    return sigmoid(z)*(1 - sigmoid(z))
end

function tanhPrime(z)
    return 1.0 - (tanh(z))^2
end

function RELU(z)
    return max(0.0, z)
end
function RELUPrime(z)
    if z >= 0
        return 1.0
    else
        return 0.0
    end
end

### feeding an input forward through the network

function evalNode(prevNodeVals, net, n)
    z = sum([w*x for (w,x) in zip(net.weights[n,1:(n-1)],prevNodeVals[1:(n-1)])]) + net.biases[n]
    return activation(z)
end

function feedForward(input, net)
    nodesToEvaluate = Array{Int32}((length(input)+1):size(net.weights,1))
    evaluatedNodes = cat(input, [0.0 for i in nodesToEvaluate], dims=1)
    
    for n in nodesToEvaluate
        evaluatedNodes[n] = evalNode(evaluatedNodes, net, n)
    end
    return evaluatedNodes   
end

function evaluateInput(input, net)
    n = size(net.weights, 1)
    i = n - net.inOutDims[2] + 1
    return feedForward(input, net)[i:n]
end


### training and backpropogation

function train(net, inputs, outputs, testInputs, testOutputs, iterations, numOfBatches, learningRate)
    trainingData = collect(zip(inputs,outputs)) # array of (input,output) pairs
    
    batchSize = length(trainingData) ÷ numOfBatches
    #println(batchSize)
    for i in 1:iterations
        println("----------------------------------------")
        println("training iteration " * string(i))
        print("batch: ")
        trainingData = shuffle(trainingData)
        for n in 1:numOfBatches
            batch = trainingData[((n-1)*batchSize)+1:n*batchSize] # (n*batchSize)-1 ?
            net = updateNet(net, batch, learningRate)
            
            print(string(n) * ", ")
            
        end
        println("")
        println(getClassificationAccuracy(net, testInputs, testOutputs))
        println(meanError(net, testInputs, testOutputs))
    end
    return net
end

function updateNet(net, batch, learningRate)
    # update net using error gradient descent

    # we find the average weight and bias gradients and then update the network
    total_dws = zeros(Float64,size(net.weights))
    total_dbs = zeros(Float64,size(net.biases))
    for (x,y) in batch
        # these are outputs of the activation function:
        nodeOutputs = feedForward(x, net)
        n = length(nodeOutputs)
        #println(n-net.inOutDims[2]+1:n)
        netOutput = nodeOutputs[n-net.inOutDims[2]+1:n]

        activationErrorDerivatives = cat(zeros(Float64, length(nodeOutputs)-net.inOutDims[2]) ,errorPartialDerivatives(netOutput,y), dims=1)
        
        # use back propogation - we start with knowing the node activations,
        # and the error derivatives with respect to the last output nodes.
        (dws,dbs) = backProp(net, nodeOutputs, activationErrorDerivatives)
        #println(sum(dws))
        total_dws += dws
        total_dbs += dbs
    end        
    
    net.weights -= learningRate * total_dws/size(batch,1)
    net.biases  -= learningRate * total_dbs/size(batch,1)
    return net
end

function backProp(net, nodeOutputs, activationErrorDerivatives)
    n = length(nodeOutputs)
    nodeGradients = zeros(Float64, n)
    wGradients = zeros(Float64, n, n)
    #bGradients = zeros(Float64, n)

    for i in 0:(n-1)
        node = n - i # iterate from last node to first (back prop...)

        nodeGradient = activationPrime(nodeOutputs[node])*activationErrorDerivatives[node]
        
        wGradients[node,:] = [ w != 0.0 ? a*nodeGradient : 0.0 for (w,a) in zip(net.weights[node,:],nodeOutputs)]
        nodeGradients[node] = nodeGradient # out bias gradient will just be 1*nodeGradients since a would = 1

        activationErrorDerivatives += (net.weights[node,:]*nodeGradient)
    end
    #
    return (wGradients,nodeGradients)
end

### error function and its derivative

function error(yNet, yTrue)
    return 0.5*sum([(yt - yn)^2 for (yt,yn) in zip(yTrue,yNet)])
end

function errorPartialDerivatives(yNet, yTrue)
    return yNet - yTrue
end




### network evaluation

function meanError(net, xs, ys)
    totalError = 0
    for (x,y) in zip(xs,ys)
        totalError += error(evaluateInput(x, net), y)
    end
    return totalError / size(xs,1)
end

function sizeOfOutputs(net, xs, ys)
    totalError = 0
    for (x,y) in zip(xs,ys)
        totalError += sum(evaluateInput(x, net))
    end
    return totalError / size(xs,1)
end


function getClassificationAccuracy(net, X_test, y_test)
    numOfTestSamples = size(X_test, 1)
    correctClassifications = 0
    for i in 1:numOfTestSamples
        if argmax(evaluateInput(X_test[i], net)) == argmax(y_test[i])
            #println("correct")
            correctClassifications += 1
        end
    end
    return (correctClassifications/numOfTestSamples)*100
end




### functions helping with testing using MNIST data 

function MNIST_image_to_feature_vector(image)
    # flatten an image to a 1d feature vector and cast to the appropriate data type
    return Array{Float64}(collect(Iterators.flatten(image)))
end

function number_value_to_out_vector(y)
    # one-hot encoding
    return Array{Float64}([ y==i ? 1.0 : 0.0 for i in 0:9])
end



#=====================
Things to do:
- handle saving network state to file so we can re load it 
- then use this to have more reliable training for much longer periods
- have training more rigorously monitor performance so we can compare different structures well

- try initializing different structures 
- try adapting a structure during training?
======================#


