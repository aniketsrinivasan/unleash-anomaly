# Semi-Supervised Anomaly Detection in Unstructured Top-K Datasets

## Usage of Anomaly Detection Tools
Follow the instructions below to get the project to function locally:
1. (To be implemented)

To run inference on given data, the following information is needed:
> * The path to the SQLite3 database containing the information;
> * The name of the table from which to extract data;
> * The name of the column containing keys (IDs) for each entry;
> * The name of the column containing values (measures) for each entry;
> * The timestamp for which inference is to be run.

All other parameters are determined by the model used, and **should remain consistent with the model**.

To run inference, initialize the model, loading from its state dictionary using 
`Model.load_state_dict(torch.load(path))`. If the model is a statistical model, information is loaded
automatically*. An example of loading a model and data is shown below:
```
# Initializing the DatasetTensor:
dataset = DatasetTensor(sqlite_path="0.topi",
                        skeleton_dict_path="stubs/skeleton_dicts/sample_skeleton_dict",
                        table_name="TOPS_80",
                        key_column="Key",
                        value_column="StatVal",
                        start_timestamp=1720809300,
                        end_timestamp=1720894800,
                        interval_size=300,
                        num_intervals=2,
                        kwargs_get_tensor=kwargs_get_tensor)
                        
# Initializing the ConvVAE model:
#   in_channels, z_channels, latent_dim, image_shape are all defined by the trained model.
VAE = ConvVAE(in_channels=2,
              z_channels=16,
              latent_dim=8,
              image_shape=(20, 20))

# Loading from state dictionary:
VAE.load_state_dict(torch.load("stubs/saved_models/sample_conv_vae.pt"))
```

Then, to run inference, the input torch.Tensor shape must match that of the trained model that is loaded. 
Indexing through a DatasetTensor returns a dictionary containing two entries: "timestamp" and "data".


An example of running inference is shown below, continuing from the above code:
```
# Getting the torch.Tensor containing data for the 0th timestep:
this_sample = dataset[0]["data"]
# Reshaping the torch.Tensor to match the size as required by the trained model:
this_sample = torch.unsqueeze(this_sample, 0)
# Running inference, where 'out' is the reconstruction of this_sample:
out, mean, logvar = VAE(this_sample)
```
After which loss can be calculated as desired (a recommendation for VAE models is to use a combination of 
mean-squared error and Kullback-Leibler divergence).

Some (statistical) models do not work using reconstruction-based methods, and instead calculate loss 
immediately, without having to explicitly define a loss function. 


## Background and Motivation on Anomaly Detection

Semi-supervised anomaly detection has been an area of research with significant progress recently.
However, many methods work with clean, well-structured data, and/or follow a supervised model training
procedure, both of which are absent in several real-world cases. One such use-case is considering
top-K (multi-variate) time-series network data; however, this is an important field to implement anomaly
detection algorithms for tasks such as premature DoS/DDoS detection.

This project uses both semi-supervised statistical methods and deep learning techniques to detect anomalies
in Unleash's top-K multi-variate time-series network datasets. 

### Brief on Structure of top-K Data

Top-K data refers to taking only the _K_ highest contributors of a given measurable value. In this case,
top-K is taken in regular time intervals for each metric, and contributors and their corresponding values
are stored as key-value pairs in SQLite3 databases. 

The block below shows a sample of data collected for a particular timestep, where _K_=20:
```
                 Key  StatVal
0              16509   213643
1              55836   139705
2              38266    21869
3              15169    21770
4              45820    18649
5              24560    16133
6              55577    15753
7              24309    11016
8             141995     7543
9               9829     6926
10            133982     4279
11             17754     3886
12             23860     2776
13            136907     2645
14            137166     1391
15             18002     1350
16            134674     1304
17            135377     1254
18            134375      980
19             45194      963
20  SYS:GROUP_TOTALS  1199096
```

This project handles such unstructured data by pre-defining an order for _N_ such keys, and storing this fixed
order as a SkeletonDict (skeleton dictionary) object. This is one of the two major underlying
datatypes used here (the other being the DataTensor, which will be explained below).

#### SkeletonDicts
A class used for handling data, intended to define a fixed order in which keys appear in arrays and other
structures. In particular, the creation of a SkeletonDict is _required_ if any other data structures in this 
project are to be used. 

SkeletonDicts are usually only loaded, and do not require re-creation unless applied to a new task.
Most functionality in data structures is implemented by either passing a SkeletonDict itself, or by providing
a skeleton dictionary path (where a pre-saved SkeletonDict can be found).

#### DataTensors
The main object of interest for the usage of Torch-based deep learning models in this project. The class uses
SkeletonDicts to convert unstructured top-K data into Numpy arrays and PyTorch Tensors. 


## Models
### Convolution-based Variational Autoencoder (ConvVAE)
### Dense-based Variational Autoencoder (DenseVAE)
### Frequency-based Statistical Modeling 
