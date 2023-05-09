This a pytorch implementation of the paper <Short-term Forecasting based on Graph Convolution Networks and Multi-resolution Convolution Neural Networks for Wind Power>.

Model training:
python main.py --args
 
where:
--save: path to save the final model 
--data: path of the data file
--num_nodes: number of variables(nodes)
--horizon: the prediction steps
--batch_size
--epochs
--device： cpu/gpu


SongYue