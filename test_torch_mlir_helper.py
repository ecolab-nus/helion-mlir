
import torch
import torch.fx as fx
from helion_fx_mlir.torch_mlir_helper import import_aten_node_to_mlir

def test_full():
    # Create a graph with aten.full
    g = fx.Graph()
    # full([10, 10], 1.0)
    node = g.call_function(torch.ops.aten.full.default, ([10, 10], 1.0))
    # We need to give it a fake output value for metadata
    node.meta['val'] = torch.zeros([10, 10])
    
    mlir = import_aten_node_to_mlir(node)
    print("--- MLIR for aten.full ---")
    print(mlir)

def test_div():
    # div(a, b)
    g = fx.Graph()
    a = g.placeholder("a")
    a.meta['val'] = torch.randn(10, 10)
    b = g.placeholder("b")
    b.meta['val'] = torch.randn(10, 10)
    
    node = g.call_function(torch.ops.aten.div.Tensor, (a, b))
    node.meta['val'] = torch.randn(10, 10)
    
    mlir = import_aten_node_to_mlir(node)
    print("\n--- MLIR for aten.div ---")
    print(mlir)

def test_list_args():
    # aten.full([d0, d1], 1.0)
    g = fx.Graph()
    d0 = g.placeholder("d0")
    d0.meta['val'] = 10
    d1 = g.placeholder("d1")
    d1.meta['val'] = 10
    
    node = g.call_function(torch.ops.aten.full.default, ([d0, d1], 1.0))
    node.meta['val'] = torch.ones(10, 10)
    
    mlir = import_aten_node_to_mlir(node)
    print("\n--- MLIR for aten.full list args ---")
    print(mlir)

if __name__ == "__main__":
    try:
        test_full()
        test_div()
        test_list_args()
    except Exception as e:
        import traceback
        traceback.print_exc()

