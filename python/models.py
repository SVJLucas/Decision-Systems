import torch
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
import gurobipy as gp # pip install gurobipy
from gurobipy import GRB
import torch.optim as optim
from abc import abstractmethod
from pmlayer.torch.layers import PMLinear # pip install pmlayer
from torch.utils.data import DataLoader, TensorDataset

class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features) # Weights cluster 1
        weights_2 = np.random.rand(num_features) # Weights cluster 2

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        u_1 = np.dot(X, self.weights[0]) # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1]) # Utility for cluster 2 = X^T.w_2
        return np.stack([u_1, u_2], axis=1) # Stacking utilities over cluster on axis 1


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n°clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.n_clusters = n_clusters
        self.n_pieces = n_pieces
        self.model, self.weights = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables."""

        model = gp.Model("TwoClusterMIP")
        weights = []

        return model, weights

    def _get_segments(self,num_criteria,num_segments,max_values,min_values):

      segments = []
      for i in range(num_criteria):
            segment_size = (max_values[i] - min_values[i]) / num_segments
            segment = [round(min_values[i] + segment_size * j, 2) for j in range(num_segments + 1)]
            segments.append(segment)
      return segments

    def find_segment_indices(self, x, segments):

      '''
      Find in which linear segment x is, so verify where:
        x in [segment[l],segment[l+1][
      Iteration notation in the intervals is l
      '''

      for l in range(len(segments) - 1):
          if x >= segments[l] and x <= segments[l + 1]:
              return l, l + 1
      return None

    def fit(self, X, Y, eps = 1e-4, M=2):

        # """Estimation of the parameters - To be completed.

        # Parameters
        # ----------
        # X: np.ndarray
        #     (n_samples, n_features) features of elements preferred to Y elements
        # Y: np.ndarray
        #     (n_samples, n_features) features of unchosen elements
        # """



        # Number of Pairs is equivalent tn the report to "P"
        number_of_pairs = len(X)

        # Getting constants
        # Here, we are going to use the same notation as in the report, which is:
        #
        #   num_criteria is equivalent tn the report to "n"
        #   num_segments is equivalent tn the report to "L"
        #   num_clusters is equivalent tn the report to "K"

        num_criteria = len(X[0])
        num_segments = self.n_pieces
        num_clusters = self.n_clusters

        # Combine corresponding columns from both matrices and find the maximum value in each combined column
        max_values = [max(max(x, y) for x, y in zip(col_x, col_y)) for col_x, col_y in zip(zip(*X), zip(*Y))]

        # Combine corresponding columns from both matrices and find the minimum value in each combined column
        min_values = [min(min(x, y) for x, y in zip(col_x, col_y)) for col_x, col_y in zip(zip(*X), zip(*Y))]

        max_values = np.ones(num_criteria)
        min_values = np.zeros(num_criteria)

        # Let's get the segments of the piecewise functions
        segments = self._get_segments(num_criteria,num_segments,max_values,min_values)


        self.segments = segments


        # Declaring errors variables

        # Overestimation error variables X
        sigma_x_plus = self.model.addMVar(number_of_pairs, lb=0, vtype='C', name=f"sigma_x_plus")

        # Underestimation error variables X
        sigma_x_minus = self.model.addMVar(number_of_pairs, lb=0, vtype='C', name=f"sigma_x_minus")

        # Overestimation error variables Y
        sigma_y_plus = self.model.addMVar(number_of_pairs, lb=0, vtype='C', name=f"sigma_y_plus")

        # Underestimation error variables Y
        sigma_y_minus = self.model.addMVar(number_of_pairs, lb=0, vtype='C', name=f"sigma_y_minus")

        # Utility function per cluster per criteria per division
        u = self.model.addMVar((num_criteria, num_segments+1, num_clusters),
                               lb=0, ub=1.0, vtype='C',
                               name=f"utility_function_per_cluster_per_criteria_per_segment")

        # Initialize utility functions for each pair and cluster without intermediate variables
        U_X = self.model.addMVar((number_of_pairs, num_clusters), lb=0, ub=1.0, vtype='C', name="utility_function_per_cluster_X")
        U_Y = self.model.addMVar((number_of_pairs, num_clusters), lb=0, ub=1.0, vtype='C', name="utility_function_per_cluster_Y")

        for j in range(number_of_pairs):
            for k in range(num_clusters):
                # For each pair and cluster, initialize an expression for the utility sum
                U_X_sum = gp.LinExpr()
                U_Y_sum = gp.LinExpr()

                for i in range(num_criteria):
                    segment_start, segment_end = self.find_segment_indices(X[j][i], segments[i])
                    # Calculate the utility directly within the constraint expression for X
                    U_X_contribution = u[i, segment_start, k] + ((X[j][i] - segments[i][segment_start]) / (segments[i][segment_end] - segments[i][segment_start])) * (u[i, segment_end, k] - u[i, segment_start, k])
                    U_X_sum += U_X_contribution

                    segment_start, segment_end = self.find_segment_indices(Y[j][i], segments[i])
                    # Calculate the utility directly within the constraint expression for Y
                    U_Y_contribution = u[i, segment_start, k] + ((Y[j][i] - segments[i][segment_start]) / (segments[i][segment_end] - segments[i][segment_start])) * (u[i, segment_end, k] - u[i, segment_start, k])
                    U_Y_sum += U_Y_contribution

                # Add constraints that define U_X and U_Y as the sum of their contributions plus errors
                self.model.addConstr(U_X[j, k] == U_X_sum + sigma_x_minus[j] - sigma_x_plus[j])
                self.model.addConstr(U_Y[j, k] == U_Y_sum + sigma_y_minus[j] - sigma_y_plus[j])

        # Monotony constraints
        for i in range(num_criteria):
            for l in range(num_segments):
                for k in range(num_clusters):
                    self.model.addConstr(u[i, l+1, k] >= u[i, l, k] + eps, f"monotony_{i}_{l}_{k}")

        # Normalization constraints
        for k in range(num_clusters):
            self.model.addConstr(gp.quicksum(u[i, num_segments, k] for i in range(num_criteria)) == 1, f"normalization_{k}")

        # Minimum of each criteria utility function must be zero
        for i in range(num_criteria):
          for k in range(num_clusters):
            self.model.addConstr(u[i,0, k]==0, f"minimum_of_each_criteria_utility_function_must_be_zero_{i}_{k}")

        # Declaration of binary variable z
        z = self.model.addVars(number_of_pairs, num_clusters, vtype = GRB.BINARY)

        # At least one client prefers x to y
        for j in range(number_of_pairs):
          self.model.addConstr(gp.quicksum(z[j,k] for k in range(num_clusters))>=1, f"at_least_one_client_prefers_x_to_y_cluster_{j}")

        # Big M method Constraint
        for j in range(number_of_pairs):
          for k in range(num_clusters):
              # If z[j,k] is 1, then u(x_j) must be greater than u(y_j), enforced by a big M method.
              self.model.addConstr((U_X[j,k] - U_Y[j,k]) >= (eps - M * (1 - z[j,k])), f"preference_{j}_{k}_X_over_Y")

        # Objective function: Minimize the sum of errors
        self.model.setObjective(gp.quicksum(sigma_x_plus[j] + sigma_x_minus[j] for j in range(number_of_pairs))+gp.quicksum(sigma_y_plus[j] + sigma_y_minus[j] for j in range(number_of_pairs)), GRB.MINIMIZE)


        # Optimize the model
        self.model.optimize()

        # Getting the Optimal Utility values
        self.utility = u.getAttr("x")


        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """

        # Number of Pairs is equivalent tn the report to "P"
        number_of_pairs = len(X)

        # Getting constants
        # Here, we are going to use the same notation as in the report, which is:
        #
        #   num_criteria is equivalent tn the report to "n"
        #   num_segments is equivalent tn the report to "L"
        #   num_clusters is equivalent tn the report to "K"

        num_criteria = len(X[0])
        num_segments = self.n_pieces
        num_clusters = self.n_clusters
        segments = self.segments

        u = self.utility
        U = np.zeros([number_of_pairs,num_clusters])

        for j in range(number_of_pairs):
            for k in range(num_clusters):
                for i in range(num_criteria):
                  segment_start, segment_end = self.find_segment_indices(X[j][i],segments[i])
                  U[j,k] += u[i,segment_start, k]  + (((X[j][i] - segments[i][segment_start])/(segments[i][segment_end]- segments[i][segment_start])) * (u[i,segment_end, k] - u[i,segment_start, k]))

        return U



class UnconstrainedEncoder(nn.Module):
    def __init__(self, num_criteria: int, num_clusters: int = 1, num_hidden_neurons: int = 128, num_hidden_layers: int = 3):
        """
        Initializes an UnconstrainedEncoder module with a configurable number of hidden layers and neurons.
        This encoder is designed for flexible, non-monotonic transformations of input data, suitable for
        a wide range of applications where strict monotonicity is not required.

        Args:
            num_criteria: Number of input features (criteria) for the encoder.
            num_clusters: Number of output dimensions for the encoder. Default is 1.
            num_hidden_neurons: Number of neurons in each hidden layer. Default is 128.
            num_hidden_layers: Number of hidden layers in the encoder. Default is 3.
        """
        # Initialize the base class
        super(UnconstrainedEncoder, self).__init__()

        # Configuration for hidden layers
        self.num_hidden_neurons = num_hidden_neurons
        self.num_hidden_layers = num_hidden_layers

        # ModuleList to hold all layers of the encoder
        self.layers = nn.ModuleList([])

        # Construct hidden layers followed by ELU activation function
        for _ in range(self.num_hidden_layers):
            # LazyLinear is used here to defer the determination of input feature size
            # until the first forward pass
            self.layers.append(nn.LazyLinear(num_hidden_neurons))
            self.layers.append(nn.ELU())  # ELU activation function for non-linear transformation

        # Output layer to map the final hidden layer's output to the desired number of clusters
        self.layers.append(nn.LazyLinear(num_clusters))
        # Sigmoid activation function to ensure output values are in the range [0, 1]
        self.layers.append(nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the UnconstrainedEncoder. Applies a series of transformations
        using the configured hidden layers and activation functions, concluding with a sigmoid activation
        to produce the final output.

        Args:
            x: A tensor of input features with shape (batch_size, num_criteria).

        Returns:
            A tensor with shape (batch_size, num_clusters) representing the encoded output,
            with each dimension's value ranging between 0 and 1.
        """
        # Sequentially apply each layer in the encoder to the input
        for layer in self.layers:
            x = layer(x)
        return x

class MonotonicEncoder(nn.Module):
    def __init__(self, num_criteria: int, num_clusters: int = 1, num_parallel_layers: int = 3):
        """
        Initializes a MonotonicEncoder module that applies multiple parallel monotonic transformations
        to the input data and averages their outputs. Designed for scenarios where maintaining
        a monotonic relationship between input features and output is crucial.

        Args:
            num_criteria: Number of input features (criteria) for the encoder.
            num_clusters: Number of output clusters. Default is 1, as the output is averaged.
            num_parallel_layers: Number of parallel layers to apply for transformation.
        """
        # Initialize the base class
        super(MonotonicEncoder, self).__init__()

        # Store initialization parameters
        self.num_criteria = num_criteria
        self.num_clusters = num_clusters
        self.num_parallel_layers = num_parallel_layers

        # Initialize a ModuleList to hold the parallel monotonic layers
        self.layers = nn.ModuleList([])
        
        # Populate the ModuleList with PMLinear layers, which are presumed to be
        # predefined classes that enforce monotonicity in the model
        for _ in range(self.num_parallel_layers):
            self.layers.append(PMLinear(num_input_dims=self.num_criteria,
                                        num_output_dims=1,
                                        indices_increasing=[i for i in range(self.num_criteria)],
                                        use_bias=True))
        
        # Sigmoid activation function to be applied to the output of each layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MonotonicEncoder. It aggregates the sigmoid-activated outputs
        of all parallel layers by averaging them.

        Args:
            x: A tensor of input features with shape (batch_size, num_criteria).

        Returns:
            A tensor with shape (batch_size, 1) representing the averaged output of all
            parallel monotonic transformations applied to the input.
        """
        # Determine the batch size from the input tensor's first dimension
        batch_size = x.size(0)
        
        # Initialize an output tensor with zeros
        out = torch.zeros((batch_size, 1), device=x.device)
        
        # Aggregate outputs from all layers
        for layer in self.layers:
            # Apply sigmoid to the layer's output and accumulate
            out += self.sigmoid(layer(x))
        
        # Average the accumulated outputs across all parallel layers
        return out / self.num_parallel_layers

class HeuristicNeuralNetwork(nn.Module):
    def __init__(self, num_criteria: int, num_clusters: int, monotonic: bool = True):
        """
        Initializes the Heuristic Neural Network with specified number of criteria and clusters.
        This network can utilize either monotonic or unconstrained encoders for each cluster.

        Args:
            num_criteria: Number of criteria/features each input example has.
            num_clusters: Number of clusters or output dimensions for utility scores.
            monotonic: Flag to use monotonic encoders if True, otherwise unconstrained encoders.
        """
        # Initialize the parent class (nn.Module)
        super(HeuristicNeuralNetwork, self).__init__()
        
        # Store the number of criteria and clusters
        self.num_criteria = num_criteria
        self.num_clusters = num_clusters
        # Flag to determine if the encoders should be monotonic
        self.monotonic = monotonic
        
        # Initialize a ModuleList to store encoders for each cluster
        self.encoders = nn.ModuleList([])
        
        # Populate the ModuleList with appropriate encoders for each cluster
        for _ in tqdm(range(self.num_clusters)):
            if self.monotonic:
                # Add a monotonic encoder for the current cluster
                self.encoders.append(MonotonicEncoder(self.num_criteria))
            else:
                # Add an unconstrained encoder for the current cluster
                self.encoders.append(UnconstrainedEncoder(self.num_criteria))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Heuristic Neural Network. It computes utility scores for each cluster
        based on the input features.

        Args:
            X: A tensor of input features with shape (batch_size, num_criteria).

        Returns:
            A tensor of utility scores with shape (batch_size, num_clusters).
        """
        # Determine the batch size from the input tensor's first dimension
        batch_size = X.size(0)
        
        # Initialize a tensor to hold utility scores, filled with zeros
        utility = torch.zeros((batch_size, self.num_clusters), device=X.device)
        
        # Iterate through each cluster's encoder to compute utility scores
        for i, encoder in enumerate(self.encoders):
            # Compute utility score for the current cluster and assign it
            # Ensures the output is appropriately sized with squeeze()
            utility[:, i] = encoder(X).squeeze()
        
        return utility

class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, num_criteria, num_clusters):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.num_criteria = num_criteria
        self.num_clusters = num_clusters
        self.model = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        model = HeuristicNeuralNetwork(self.num_criteria, self.num_clusters)
        return model

    def _calculate_preference_loss(
        self,
        utilities_product_x: torch.Tensor,
        utilities_product_y: torch.Tensor,
        predicted_utility_min: torch.Tensor,
        predicted_utility_max: torch.Tensor,
        weight_preference: float = 1.0,
        weight_min_utility_deviation: float = 10.0,
        weight_max_utility_deviation: float = 10.0
    ) -> torch.Tensor:
        """
        Calculates the loss to enforce a single-dimensional preference between two products by penalizing deviations
        from ideal utility predictions for products at minimum and maximum criteria values. It aims to ensure
        that the utility difference in one dimension favors product X over product Y, while penalizing any
        deviation from expected utility values when all criteria are at their minimum or maximum.

        Args:
            utilities_product_x: Tensor of utilities for product X.
            utilities_product_y: Tensor of utilities for product Y.
            predicted_utility_min: Tensor of predicted utilities for product with minimum criteria.
            predicted_utility_max: Tensor of predicted utilities for product with maximum criteria.
            weight_preference: Weight for the preference loss component.
            weight_min_utility_deviation: Weight for the penalty on deviation from minimum utility prediction.
            weight_max_utility_deviation: Weight for the penalty on deviation from maximum utility prediction.

        Returns:
            A tensor representing the calculated loss, emphasizing single-dimensional preference and penalizing
            deviations from ideal utility predictions.
        """

        # Calculate differences in utilities between product X and Y
        utility_differences = utilities_product_x - utilities_product_y

        # Calculate basic preference loss using the mean of product-wise ReLU activations of utility differences
        basic_preference_loss = torch.mean(torch.prod(torch.relu(utility_differences), dim=1))

        # Encourage utility differences to be negative in exactly one dimension
        negative_differences = (utility_differences < 0).float()  # Convert boolean to float
        exact_negative_difference_loss = torch.sum(((torch.sum(negative_differences, dim=1) - 1) ** 2))
        
        # Weighted sum of preference losses
        weighted_preference_loss = (exact_negative_difference_loss * weight_preference * 100) + \
                                  (basic_preference_loss * weight_preference / 10)

        # Compute losses for deviations from ideal predictions at min and max criteria
        loss_min_criteria_deviation = torch.sum((predicted_utility_min - torch.zeros_like(predicted_utility_min)) ** 2)
        loss_max_criteria_deviation = torch.sum((predicted_utility_max - torch.ones_like(predicted_utility_max)) ** 2)

        # Approximate the count of negative differences using sigmoid for a softer condition
        sigmoid_approximation = torch.sigmoid(-10 * utility_differences)
        negative_difference_count_penalty = torch.square(torch.sum(sigmoid_approximation, dim=1) - 1)

        # Penalty for exact negative differences
        penalty_for_exact_neg_diff = torch.sum(torch.where(torch.sum(negative_differences, dim=1) == 1,
                                                          -torch.sum(torch.log((utility_differences) ** 2 + 1e-8), dim=1),
                                                          0))

        # Total loss calculation including preference loss and penalties for utility prediction deviations
        total_loss = weighted_preference_loss + \
                    weight_min_utility_deviation * loss_min_criteria_deviation + \
                    weight_max_utility_deviation * loss_max_criteria_deviation + \
                    penalty_for_exact_neg_diff - \
                    torch.sum(torch.log((utilities_product_x - utilities_product_y) ** 2 + 1e-8))

        return total_loss


    def _get_loader_from_dataset(self, X: np.ndarray, Y: np.ndarray, batch_size: int) -> DataLoader:
        """
        Creates a DataLoader object from numpy arrays containing the criteria values of the preferred and
        non-preferred products.

        Args:
            X (np.ndarray): Numpy array containing the criteria values of the preferred product X.
            Y (np.ndarray): Numpy array containing the criteria values of the non-preferred product Y.
            batch_size (int): Size of the batches to use in the DataLoader.

        Returns:
            DataLoader: The created DataLoader object for iterating over the dataset.
        """
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.Tensor(X)
        Y_tensor = torch.Tensor(Y)

        # Create a TensorDataset object with the tensors
        dataset = TensorDataset(X_tensor, Y_tensor)

        # Create a DataLoader object with the dataset, specifying batch size and shuffling
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return loader

    def _pe_score(self, utilities_x: torch.Tensor, utilities_y: torch.Tensor) -> float:
        """
        Calculates the Performance Evaluation (PE) score between two sets of utilities, `utilities_x`
        and `utilities_y`. The PE score represents the proportion of instances where `utilities_x`
        is not strictly preferred over `utilities_y` across all dimensions. A higher PE score indicates
        a higher frequency of `utilities_x` being preferred, with a score of 1 meaning `utilities_x`
        is always preferred, and a score close to 0 meaning rarely preferred.

        Args:
            utilities_x: A tensor containing the utilities of items in set X with shape (batch_size, num_criteria).
            utilities_y: A tensor containing the utilities of items in set Y with shape (batch_size, num_criteria).

        Returns:
            The PE score as a float, which is the normalized count of instances where `utilities_x` is not
            strictly preferred over `utilities_y` across all evaluated dimensions.

        Note:
            The method computes the product of ReLU activations of the difference between `utilities_x` and
            `utilities_y` along the criteria dimension. If the product is greater than 0, it implies that
            `utilities_x` is strictly preferred over `utilities_y` across all dimensions. The PE score is
            then calculated as the proportion of instances not meeting this condition.
        """
        # Determine the batch size from utilities_x tensor
        batch_size = utilities_x.size(0)
        
        # Calculate the condition where utilities_x is strictly preferred over utilities_y
        # across all dimensions. The result is a binary tensor where 1 indicates preference
        # of utilities_x over utilities_y in all dimensions, and 0 otherwise.
        preference_indicator = torch.where(torch.prod(torch.relu(utilities_x - utilities_y), dim=1) > 0, 1, 0)
        
        # Compute the PE score by subtracting the sum of the preference indicators from the batch size
        # and then normalizing by the batch size. This gives the proportion of instances where utilities_x
        # is not strictly preferred over utilities_y.
        pe = (batch_size - torch.sum(preference_indicator)) / batch_size
        
        # Convert the PE score to a float for easy interpretation and return
        pe = float(pe)
        return pe

    def _train_final_model(self, model: nn.Module, total_training_loader: DataLoader, n_epochs: int, optimizer: optim.Optimizer, criterion: nn.Module):

        """
        Conducts the training loop for the specified PyTorch model over a given number of epochs,
        utilizing a specified optimizer and loss function. This function is designed to process batches
        of data, compute the loss, update model parameters, and track the training progress by recording
        the average loss and performance evaluation (PE) score per epoch.

        Args:
            model (nn.Module): The PyTorch model that will be trained. This should be an instance
                of a class derived from nn.Module, prepared for training.
            total_training_loader (DataLoader): DataLoader instance providing batches of training
                data. It should yield two tensors (features and targets) per iteration.
            n_epochs (int): The total number of training epochs. An epoch is a single pass through
                the entire training dataset.
            optimizer (optim.Optimizer): The optimizer used for adjusting the model parameters
                based on gradients to minimize the loss function. Examples include Adam, SGD, etc.
            criterion (nn.Module): The loss function used to evaluate the model's predictions against
                the actual targets. It guides the optimizer by indicating how well the model is performing.

        Returns:
            mean_loss_train (list): A list containing the mean loss for the training set calculated
                at the end of each epoch, providing insight into how the model's performance improves.
            mean_pe_train (list): A list containing the mean Performance Evaluation (PE) score for
                the training set calculated at the end of each epoch. This metric indicates the model's
                predictive accuracy or the consistency of its predictions with the desired outcomes.

        Note:
            The actual computation of the PE score should be adapted based on specific requirements
            and might involve additional logic not shown here. This function assumes the calculation
            is integrated within the training loop or the criterion.
        """

        # initialize lists to keep track of metrics
        mean_loss_train = []
        mean_pe_train = []

        one = torch.ones((1,model.num_criteria))
        zero = torch.zeros((1,model.num_criteria))

        # loop over epochs
        for it in range(n_epochs):

            # initialize list to keep track of train loss for this epoch
            train_loss = []
            train_pe = []

            # set model to train mode
            model.train()

            # loop over training data
            for X, Y in total_training_loader:

                # zero the gradients
                optimizer.zero_grad()

                # forward pass
                utilities_x = model(X)
                utilities_y = model(Y)
                zero_pred = model(zero)
                one_pred = model(one)

                # compute loss
                loss = criterion(utilities_x, utilities_y,zero_pred,one_pred)

                # backward pass and optimization step
                loss.backward()
                optimizer.step()

                # append loss to train_loss list
                train_loss.append(loss.item())

                # Explained Pairs Score
                train_pe.append(self._pe_score(utilities_x,utilities_y))

            # append the mean train loss for this epoch to the list of train losses
            mean_loss_train.append(np.mean(train_loss))
            mean_pe_train.append(np.mean(train_pe))

            # print epoch metrics
            print(f'Epoch {it}/{n_epochs}, Train Loss: {mean_loss_train[-1]:.4f}, Train pe: {mean_pe_train[-1]:.4f}')

        # return lists of mean train
        return mean_loss_train, mean_pe_train

    def fit(self, X: np.ndarray, Y: np.ndarray, learning_rate: float = 1e-2, batch_size: int = 256, n_epochs: int = 100):
        """
        Trains the model using the provided datasets X and Y, where X contains features of elements preferred over those in Y.
        The method applies a preference learning approach to estimate the model parameters, optimizing a custom loss function
        that quantifies single dimension preference discrepancies between pairs of items in X and Y.

        Parameters
        ----------
        X : np.ndarray
            An array of shape (n_samples, n_features) representing the features of elements preferred to those in Y.
        Y : np.ndarray
            An array of shape (n_samples, n_features) representing the features of unchosen (non-preferred) elements.
        learning_rate : float, optional
            The learning rate for the optimizer. Defaults to 1e-2.
        batch_size : int, optional
            The size of batches for training the model. Defaults to 256.
        n_epochs : int, optional
            The number of epochs to train the model. Defaults to 200.

        Returns
        -------
        self
            Returns an instance of the class with the model parameters updated through training.

        Overview
        --------
        The training process involves:
        - Creating a data loader from the datasets X and Y for batch-wise processing.
        - Setting up a custom criterion function for calculating the preference loss.
        - Initializing an optimizer with the model parameters and the specified learning rate.
        - Iteratively training the model across the specified number of epochs while tracking the average loss and performance evaluation (PE) score.
        """

        # Initialize the data loader for batch processing of datasets X and Y
        train_loader = self._get_loader_from_dataset(X, Y, batch_size)
        
        # Set the custom criterion for calculating preference loss
        criterion = self._calculate_preference_loss
        
        # Initialize the optimizer with the AdamW algorithm for model parameters
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Train the model and capture the mean training loss and PE score over epochs
        self.mean_loss_train, self.mean_pe_train = self._train_final_model(
            self.model,
            train_loader,
            n_epochs=n_epochs,
            optimizer=optimizer,
            criterion=criterion
        )
        
        # Return the instance of the class after training
        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """

        utility = self.model(torch.Tensor(X))
        utility = utility.cpu().detach().numpy()
        
        return utility
