import itertools
import torch

def enumerate_paths(n, n_branches):
    """
    Enumerate all paths (from the root to the n-th decision node) in a perfect 
    balanced k-ary tree with a fixed number of decisions (n) and fixed number 
    of branches (n_branches).

    The tree is assumed to be stored in "heap order":
       - The root is at index 0.
       - For a node at index i, its b-th child (with b in {0, ..., n_branches-1}) 
         is at index: n_branches * i + (b + 1).

    Parameters:
      n (int): Number of decision layers (i.e., the path length).
      n_branches (int): Number of branches (children) per node (>=2).

    Returns:
      node_paths (torch.LongTensor): Tensor of shape (n_all_paths, n) with the indices 
                                     of nodes where decisions occur.
      branch_paths (torch.LongTensor): Tensor of shape (n_all_paths, n) with the branch 
                                       choices made at each corresponding node.
    """
    # Generate all possible sequences of branch choices, where each sequence is of length n.
    branch_sequences = list(itertools.product(range(n_branches), repeat=n))
    node_paths = []
    branch_paths = []
    
    # For each branch choice sequence, compute the corresponding node indices.
    for seq in branch_sequences:
        nodes = [0]  # The first decision is always at the root (index 0).
        # For each decision (except the last one, because we only need the node where the decision is made)
        # we compute the next node index.
        for i in range(1, n):
            current_node = nodes[i-1]
            branch_choice = seq[i-1]
            next_node = n_branches * current_node + (branch_choice + 1)
            nodes.append(next_node)
        # Append the list of node indices and branch choices for this path.
        node_paths.append(nodes)
        branch_paths.append(list(seq))
    
    # Convert the lists to PyTorch tensors.
    node_paths = torch.tensor(node_paths, dtype=torch.long)    # shape: (n_all_paths, n)
    branch_paths = torch.tensor(branch_paths, dtype=torch.long)  # shape: (n_all_paths, n)
    expert_indices = node_paths * n_branches + branch_paths # shape: (n_all_paths, n)
    
    return node_paths, branch_paths, expert_indices