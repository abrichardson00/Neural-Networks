using Random
using MLDatasets
using LightGraphs, SimpleWeightedGraphs
using GraphPlot
using Compose
using Cairo
include("./NN.jl")


### Training network on MNIST digits
#= 

# load full training set
println("loading data...")
train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

X_train = [MNIST_image_to_feature_vector(train_x[:,:,i]) for i in 1:size(train_x,3)]
y_train = [number_value_to_out_vector(y) for y in train_y]


X_test = [MNIST_image_to_feature_vector(test_x[:,:,i]) for i in 1:size(test_x,3)]
y_test = [number_value_to_out_vector(y) for y in test_y]

netDimensions = [784,200,80,10]
net = Net(initWeights(netDimensions), initBiases(netDimensions), [784,10])
# for randomized network structure, use this instead:
# net = Net(initWeightsRandom([784,10], 200, 0.5), [(rand()*2.0)-1.0 for i in 1:(784+200+10)], [784,10])

trainedNet = train(net, X_train[1:20000], y_train[1:20000], 4, 500)

=#


### Creating some made up simpler inputs and outputs to test the network training
#=

trainingInputs = [[rand() for j in 1:10] for i in 1:200]
trainingOutputs = [[sigmoid(trainingInputs[i][1] - trainingInputs[i][2]), sigmoid(trainingInputs[i][3]*3.0 + trainingInputs[i][4]), sigmoid(sum(trainingInputs[i][5:end]))] for i in 1:size(trainingInputs, 1)]
testInputs = [[rand() for j in 1:10] for i in 1:100]
testOutputs = [[sigmoid(testInputs[i][1] - testInputs[i][2]), sigmoid(testInputs[i][3]*3.0 + testInputs[i][4]), sigmoid(sum(testInputs[i][5:end]))] for i in 1:size(testInputs, 1)]

ioDims = [10,3]
#testNet = Net(initWeightsRandom(ioDims, 20, 0.3), [(rand()*2.0)-1.0 for i in 1:(10 + 20 + 3)], ioDims) # for random structure
testNet = Net(initWeights([10,10,5,3]), initBiases([10,10,5,3]), ioDims)
testNetTrained = train(testNet, trainingInputs, trainingOutputs, testInputs, testOutputs, 100, 20, 0.1)
=#


### Creating any random network we want
# we displaying the adjacency matrix of the graph
# and then create an image of the graph

testNet = Net(initWeights([50,20,5]), initBiases([50,20,5]), [50,5])

printAdjacencyMatrix(testNet)

g = SimpleWeightedDiGraph(transpose(testNet.weights))
a = context()
out = compose(a, Compose.rectangle() ,fill("black"))

nodeColours = []
N = length(LightGraphs.vertices(g))
for i in 1:N
    if i <= testNet.inOutDims[1]
        append!(nodeColours, ["wheat"])
    elseif (N - i) < testNet.inOutDims[2]
        append!(nodeColours, ["tomato"])
    else
        append!(nodeColours, ["turquoise"])
    end
end

draw(PNG("testNet.png", 50cm, 50cm), compose(out, gplot(g, layout=circular_layout,nodefillc=nodeColours, arrowlengthfrac=0.02)))