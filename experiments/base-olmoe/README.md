## General Principles 
- The default code only works for a single node (one process) with any amount of GPUs attached to that process. Multi-node is not supported as it add significant additional complexity, and for experimental purposes it's unlikely need multiple nodes/processes anyways.
- The suggested workflow has a lot of "copy and pasting" in order to reduce abstraction and maintain ease of rapid experimentation.

## Suggested Workflow
1. Create a working model class and test that basic training speed/memory usage are acceptable.
- `initial_test.ipynb` contains an example.