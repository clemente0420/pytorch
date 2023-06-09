from typing import NamedTuple, Callable, Any, Tuple, List, Dict, Type, cast, Optional, TypeVar, overload, Union
import functools
from collections import namedtuple, OrderedDict
from dataclasses import dataclass


T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
R = TypeVar('R')

"""
包含用于处理嵌套 Python 数据结构的实用函数。
*pytree* 是 Python 的嵌套数据结构。从树的角度来看，节点是 Python 的集合（例如列表、元组、字典），叶子是 Python 的值。此外，pytree 不应包含引用循环。
pytree 对于处理嵌套的张量集合很有用。例如，可以使用 `tree_map` 在某个嵌套的张量集合上映射函数，使用 `tree_unflatten` 获取嵌套集合中所有张量的扁平列表。pytree 对于实现 PyTorch API 的嵌套集合支持非常有帮助。
由于 Python 开销，这种 pytree 实现性能不佳。
为了提高性能，可以将部分实现移到 C++ 中。

Contains utility functions for working with nested python data structures.

A *pytree* is Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_unflatten` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.

This pytree implementation is not very performant due to Python overhead
To improve the performance we can move parts of the implementation to C++.
"""

# A NodeDef holds two callables:
# - flatten_fn should take the collection and return a flat list of values.
#   It can also return some context that is used in reconstructing the
#   collection.
# - unflatten_fn should take a flat list of values and some context
#   (returned by flatten_fn). It returns the collection by reconstructing
#   it from the list and the context.
# - to_str_fn takes a TreeSpec with the specific type and a list of its children
#   TreeSpecs already converted to strings, and returns a string representation
#   of this TreeSpec
# - maybe_from_str_fn takes in a string and if this string represents a TreeSpec
#   of this type, returns the type, the context, and a string representation of
#   its children specs. Otherwise it returns None.

# Context 是一个占位符类型，表示上下文信息，可以是任何类型的值。
Context = Any

# PyTree 是一个占位符类型，表示 Python 的嵌套数据结构。
# 在树的角度来看，节点是 Python 的集合（例如列表、元组、字典），
# 叶子是 Python 的值。
# PyTree 不应包含引用循环。
PyTree = Any

# FlattenFunc 是一个函数类型，接受 PyTree 并返回一个包含平坦值列表和上下文信息的元组。
FlattenFunc = Callable[[PyTree], Tuple[List, Context]]

# UnflattenFunc 是一个函数类型，接受平坦值列表和上下文信息，并通过从列表和上下文中重构集合来返回 PyTree。
UnflattenFunc = Callable[[List, Context], PyTree]

# ToStrFunc 是一个函数类型，接受具有特定类型的 TreeSpec 和已转换为字符串的其子 TreeSpec 的列表，
# 并返回 TreeSpec 的字符串表示形式。
ToStrFunc = Callable[["TreeSpec", List[str]], str]

# MaybeFromStrFunc 是一个函数类型，接受字符串作为输入。
# 如果该字符串表示该类型的 TreeSpec，则返回类型、上下文和其子规范的字符串表示形式。
# 否则返回 None。
MaybeFromStrFunc = Callable[[str], Optional[Tuple[Any, Context, str]]]

class NodeDef(NamedTuple):
    type: Type[Any]  # 类型
    flatten_fn: FlattenFunc  # 平铺函数
    unflatten_fn: UnflattenFunc  # 反平铺函数
    to_str_fn: ToStrFunc  # 转换为字符串函数
    maybe_from_str_fn: MaybeFromStrFunc  # 从字符串中恢复函数

SUPPORTED_NODES: Dict[Type[Any], NodeDef] = {}  # 支持的节点字典，用于存储不同类型的 NodeDef 对象

def _register_pytree_node(
    typ: Any,
    flatten_fn: FlattenFunc,
    unflatten_fn: UnflattenFunc,
    to_str_fn: Optional[ToStrFunc] = None,
    maybe_from_str_fn: Optional[MaybeFromStrFunc] = None,
) -> None:
    def _raise_error(_):  # type: ignore[no-untyped-def]
        raise NotImplementedError(f"Serializing {typ} not implemented")
    if to_str_fn is None:
        to_str_fn = _raise_error   # type: ignore[assignment, return-value]
    if maybe_from_str_fn is None:
        maybe_from_str_fn = _raise_error  # type: ignore[assignment, return-value]
    assert to_str_fn is not None
    assert maybe_from_str_fn is not None
    node_def = NodeDef(typ, flatten_fn, unflatten_fn, to_str_fn, maybe_from_str_fn)
    SUPPORTED_NODES[typ] = node_def

def _str_to_dict(str_spec: str) -> Tuple[List[str], str]:
    assert str_spec[1] == "("  # 确保字符串以 "(" 开始
    assert str_spec[-1] == ")"  # 确保字符串以 ")" 结束
    context_and_child_strings = str_spec[2:-1]  # 获取上下文和子字符串的部分

    child_strings = []  # 子字符串列表
    context_strings = []  # 上下文字符串列表
    nested_parentheses = 0  # 嵌套括号计数
    start_index = 0  # 子字符串的起始索引
    for i, char in enumerate(context_and_child_strings):
        if char == ":":
            if nested_parentheses == 0:
                context_strings.append(context_and_child_strings[start_index:i])  # 添加上下文字符串
                start_index = i + 1
        elif char == "(":
            nested_parentheses += 1
        elif char == ")":
            nested_parentheses -= 1

        if nested_parentheses == 0 and char == ",":
            child_strings.append(context_and_child_strings[start_index:i])  # 添加子字符串
            start_index = i + 1

    child_strings.append(context_and_child_strings[start_index:])
    return context_strings, ','.join(child_strings)  # 返回上下文字符串列表和子字符串拼接后的字符串

def _dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())  # 将字典的值和键分别转换为列表，并返回

def _dict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return dict(zip(context, values))  # 使用键和值列表构建字典，并返回

def _dict_to_str(spec: "TreeSpec", child_strings: List[str]) -> str:
    assert spec.type == dict  # 确保类型为字典
    context_child_strings = []
    for key, child_string in zip(spec.context, child_strings):
        context_child_strings.append(f"{key}:{child_string}")  # 将键和子字符串拼接为上下文-子字符串对
    return f"D({','.join(context_child_strings)})"  # 返回形如 D(...) 的字符串表示

def _maybe_str_to_dict(str_spec: str) -> Optional[Tuple[Any, Context, str]]:
    if not str_spec.startswith("D"):
        return None
    context_strings, child_strings = _str_to_dict(str_spec)
    return dict, context_strings, child_strings  # 返回字典类型、上下文字符串列表和子字符串列表的元组

def _list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    return d, None  # 列表已经是平铺状态，直接返回列表和 None

def _list_unflatten(values: List[Any], context: Context) -> List[Any]:
    return list(values)  # 返回列表本身

def _list_to_str(spec: "TreeSpec", child_strings: List[str]) -> str:
    assert spec.type == list  # 确保类型为列表
    return f"L({','.join(child_strings)})"  # 返回形如 L(...) 的字符串表示


def _maybe_str_to_list(str_spec: str) -> Optional[Tuple[Any, Context, str]]:
    if not str_spec.startswith("L"):
        return None
    assert str_spec[1] == "("
    assert str_spec[-1] == ")"
    children_string = str_spec[2:-1]
    return list, None, children_string  # 返回列表类型、None 作为上下文、子字符串的元组

def _tuple_flatten(d: Tuple[Any, ...]) -> Tuple[List[Any], Context]:
    return list(d), None  # 将元组转换为列表形式，并返回列表和 None

def _tuple_unflatten(values: List[Any], context: Context) -> Tuple[Any, ...]:
    return tuple(values)  # 返回元组形式的列表

def _tuple_to_str(spec: "TreeSpec", child_strings: List[str]) -> str:
    assert spec.type == tuple  # 确保类型为元组
    return f"T({','.join(child_strings)})"  # 返回形如 T(...) 的字符串表示

def _maybe_str_to_tuple(str_spec: str) -> Optional[Tuple[Any, Context, str]]:
    if not str_spec.startswith("T"):
        return None
    assert str_spec[1] == "("
    assert str_spec[-1] == ")"
    children_string = str_spec[2:-1]
    return tuple, None, children_string  # 返回元组类型、None 作为上下文、子字符串的元组

def _namedtuple_flatten(d: NamedTuple) -> Tuple[List[Any], Context]:
    return list(d), type(d)  # 将命名元组转换为列表形式，并返回列表和命名元组的类型

def _namedtuple_unflatten(values: List[Any], context: Context) -> NamedTuple:
    return cast(NamedTuple, context(*values))  # 根据命名元组的类型和值列表构建命名元组，并返回

def _namedtuple_to_str(spec: "TreeSpec", child_strings: List[str]) -> str:
    assert spec.type == namedtuple  # 确保类型为命名元组
    context_type = {spec.context.__name__}
    context_fields = str(spec.context._fields).replace("'", "")
    context_type = spec.context.__name__
    return f"N({context_type}{context_fields},{','.join(child_strings)})"  # 返回形如 N(...) 的字符串表示


def _maybe_str_to_namedtuple(str_spec: str) -> Optional[Tuple[Any, Context, str]]:
    if not str_spec.startswith("N"):
        return None
    assert str_spec[1] == "("
    assert str_spec[-1] == ")"
    context_end_idx = str_spec.find(")") + 1
    context_str = str_spec[2:context_end_idx]
    children_string = str_spec[context_end_idx + 1:-1]

    # 创建上下文命名元组
    type_end_idx = context_str.find("(")
    context_type_str = context_str[:type_end_idx]
    assert context_str[-1] == ")"
    namedtuple_fields_str = context_str[type_end_idx + 1:-1]
    context = namedtuple(context_type_str, namedtuple_fields_str)  # type: ignore[misc]

    return namedtuple, context, children_string  # 返回命名元组类型、上下文、子字符串的元组

def _odict_flatten(d: 'OrderedDict[Any, Any]') -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())  # 将有序字典转换为值和键的列表形式，并返回列表和键的列表作为上下文

def _odict_unflatten(values: List[Any], context: Context) -> 'OrderedDict[Any, Any]':
    return OrderedDict((key, value) for key, value in zip(context, values))  # 根据上下文和值的列表重新构建有序字典，并返回

def _odict_to_str(spec: "TreeSpec", child_strings: List[str]) -> str:
    assert spec.type == OrderedDict  # 确保类型为有序字典
    context_child_strings = []
    for key, child_string in zip(spec.context, child_strings):
        context_child_strings.append(f"{key}:{child_string}")
    return f"O({','.join(context_child_strings)})"  # 返回形如 O(...) 的字符串表示

def _maybe_str_to_odict(str_spec: str) -> Optional[Tuple[Any, Context, str]]:
    if not str_spec.startswith("O"):
        return None
    context_strings, child_strings = _str_to_dict(str_spec)
    return OrderedDict, context_strings, child_strings  # 返回有序字典类型、上下文字符串列表、子字符串的元组


_register_pytree_node(dict, _dict_flatten, _dict_unflatten, _dict_to_str, _maybe_str_to_dict)
# 注册字典类型，指定相应的 flatten、unflatten、to_str 和 maybe_from_str 函数

_register_pytree_node(list, _list_flatten, _list_unflatten, _list_to_str, _maybe_str_to_list)
# 注册列表类型，指定相应的 flatten、unflatten、to_str 和 maybe_from_str 函数

_register_pytree_node(tuple, _tuple_flatten, _tuple_unflatten, _tuple_to_str, _maybe_str_to_tuple)
# 注册元组类型，指定相应的 flatten、unflatten、to_str 和 maybe_from_str 函数

_register_pytree_node(namedtuple, _namedtuple_flatten, _namedtuple_unflatten, _namedtuple_to_str, _maybe_str_to_namedtuple)
# 注册命名元组类型，指定相应的 flatten、unflatten、to_str 和 maybe_from_str 函数

_register_pytree_node(OrderedDict, _odict_flatten, _odict_unflatten, _odict_to_str, _maybe_str_to_odict)
# 注册有序字典类型，指定相应的 flatten、unflatten、to_str 和 maybe_from_str 函数

# h/t https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
def _is_namedtuple_instance(pytree: Any) -> bool:
    typ = type(pytree)
    bases = typ.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(typ, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(type(entry) == str for entry in fields)
# 检查一个对象是否是命名元组实例的函数
# 该函数检查对象的类型是否是元组的子类，且具有 "_fields" 属性，且 "_fields" 属性的值为元组且元组中的每个元素都是字符串

def _get_node_type(pytree: Any) -> Any:
    if _is_namedtuple_instance(pytree):
        return namedtuple
    return type(pytree)
# 获取给定 PyTree 对象的节点类型的函数
# 如果给定对象是命名元组实例，则返回 namedtuple，否则返回对象的类型

# A leaf is defined as anything that is not a Node.
def _is_leaf(pytree: PyTree) -> bool:
    return _get_node_type(pytree) not in SUPPORTED_NODES
# 判断一个 PyTree 对象是否为叶子节点的函数
# 如果对象的节点类型不在支持的节点类型字典 SUPPORTED_NODES 中，则认为该对象是叶子节点


# A TreeSpec represents the structure of a pytree. It holds:
# "type": the type of root Node of the pytree
# context: some context that is useful in unflattening the pytree
# children_specs: specs for each child of the root Node
# num_leaves: the number of leaves

# TreeSpec 表示一个 pytree 的结构。它包含以下信息：
# "type": pytree 根节点的类型
# context: 在还原 pytree 时有用的一些上下文信息
# children_specs: 根节点的每个子节点的规格
# num_leaves: 叶子节点的数量

@dataclass
class TreeSpec:
    type: Any  # 根节点的类型
    context: Context  # 还原 pytree 所需的上下文信息
    children_specs: List['TreeSpec']  # 根节点的每个子节点的规格

    def __post_init__(self) -> None:
        self.num_leaves: int = sum([spec.num_leaves for spec in self.children_specs])  # 叶子节点的数量

    def __repr__(self, indent: int = 0) -> str:
        repr_prefix: str = f'TreeSpec({self.type.__name__}, {self.context}, ['
        children_specs_str: str = ''
        if len(self.children_specs):
            indent += len(repr_prefix)
            children_specs_str += self.children_specs[0].__repr__(indent)
            children_specs_str += ',' if len(self.children_specs) > 1 else ''
            children_specs_str += ','.join(['\n' + ' ' * indent + child.__repr__(indent) for child in self.children_specs[1:]])
        repr_suffix: str = f'{children_specs_str}])'
        return repr_prefix + repr_suffix


class LeafSpec(TreeSpec):
    def __init__(self) -> None:
        super().__init__(None, None, [])
        self.num_leaves = 1

    def __repr__(self, indent: int = 0) -> str:
        return '*'  # 表示叶子节点

def tree_flatten(pytree: PyTree) -> Tuple[List[Any], TreeSpec]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used
    to reconstruct the pytree.
    """
    if _is_leaf(pytree):  # 如果是叶子节点
        return [pytree], LeafSpec()

    node_type = _get_node_type(pytree)  # 获取节点类型
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn  # 获取节点类型对应的 flatten 函数
    child_pytrees, context = flatten_fn(pytree)  # 调用 flatten 函数得到子节点列表和上下文信息

    # 递归地对子节点进行 flatten
    result: List[Any] = []
    children_specs: List['TreeSpec'] = []
    for child in child_pytrees:
        flat, child_spec = tree_flatten(child)  # 递归调用 tree_flatten 函数
        result += flat
        children_specs.append(child_spec)

    return result, TreeSpec(node_type, context, children_specs)


def tree_unflatten(values: List[Any], spec: TreeSpec) -> PyTree:
    """给定一个值列表和一个 TreeSpec，构建一个 pytree。
    这是 `tree_flatten` 的逆操作。
    """
    if not isinstance(spec, TreeSpec):
        raise ValueError(
            f'tree_unflatten(values, spec): 期望 `spec` 是 TreeSpec 类型，但得到的是 {type(spec)} 类型的对象。')
    if len(values) != spec.num_leaves:
        raise ValueError(
            f'tree_unflatten(values, spec): `values` 的长度为 {len(values)}，'
            f'但 spec 表示的 pytree 包含 {spec.num_leaves} 个元素 ({spec})。')
    if isinstance(spec, LeafSpec):
        return values[0]

    unflatten_fn = SUPPORTED_NODES[spec.type].unflatten_fn

    # 递归地还原子节点
    start = 0
    end = 0
    child_pytrees = []
    for child_spec in spec.children_specs:
        end += child_spec.num_leaves
        child_pytrees.append(tree_unflatten(values[start:end], child_spec))
        start = end

    return unflatten_fn(child_pytrees, spec.context)


def tree_map(fn: Any, pytree: PyTree) -> PyTree:
    """对给定的 pytree 进行映射操作。

    参数：
        - fn: 要应用于每个元素的函数。
        - pytree: 要映射的 pytree 对象。

    返回值：
        应用映射函数后得到的新的 pytree 对象。
    """
    flat_args, spec = tree_flatten(pytree)
    return tree_unflatten([fn(i) for i in flat_args], spec)

Type2 = Tuple[Type[T], Type[S]]
Type3 = Tuple[Type[T], Type[S], Type[U]]
TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]

Fn3 = Callable[[Union[T, S, U]], R]
Fn2 = Callable[[Union[T, S]], R]
Fn = Callable[[T], R]
FnAny = Callable[[Any], R]

MapOnlyFn = Callable[[T], Callable[[Any], Any]]

@overload
def map_only(ty: Type2[T, S]) -> MapOnlyFn[Fn2[T, S, Any]]:
    ...

@overload
def map_only(ty: Type[T]) -> MapOnlyFn[Fn[T, Any]]:
    ...

@overload
def map_only(ty: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    ...

def map_only(ty: TypeAny) -> MapOnlyFn[FnAny[Any]]:
    """
    假设您正在编写一个只对张量进行 tree_map 操作，而不改变其他元素的函数。通常情况下，您需要编写如下代码：

        def go(t):
            if isinstance(t, Tensor):
                return ...
            else:
                return t

    使用该函数，您只需要编写如下代码：

        @map_only(Tensor)
        def go(t):
            return ...

    您还可以直接使用 'tree_map_only' 函数。
    """
    def deco(f: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(f)
        def inner(x: T) -> Any:
            if isinstance(x, ty):
                return f(x)
            else:
                return x
        return inner
    return deco

@overload
def tree_map_only(ty: Type[T], fn: Fn[T, Any], pytree: PyTree) -> PyTree:
    ...

@overload
def tree_map_only(ty: Type2[T, S], fn: Fn2[T, S, Any], pytree: PyTree) -> PyTree:
    ...

@overload
def tree_map_only(ty: Type3[T, S, U], fn: Fn3[T, S, U, Any], pytree: PyTree) -> PyTree:
    ...

def tree_map_only(ty: TypeAny, fn: FnAny[Any], pytree: PyTree) -> PyTree:
    """
    对给定的 pytree 进行映射操作，只对特定类型的元素应用映射函数。

    参数：
        - ty: 要应用映射函数的元素类型。
        - fn: 要应用的映射函数。
        - pytree: 要映射的 pytree 对象。

    返回值：
        应用映射函数后得到的新的 pytree 对象。
    """
    return tree_map(map_only(ty)(fn), pytree)

def tree_all(pred: Callable[[Any], bool], pytree: PyTree) -> bool:
    """检查 pytree 中的所有元素是否都满足给定的条件。

    参数：
        - pred: 要检查的条件函数，接受一个元素并返回布尔值。
        - pytree: 要检查的 pytree 对象。

    返回值：
        如果 pytree 中的所有元素都满足条件，则返回 True，否则返回 False。
    """
    flat_args, _ = tree_flatten(pytree)
    return all(map(pred, flat_args))

def tree_any(pred: Callable[[Any], bool], pytree: PyTree) -> bool:
    """检查 pytree 中是否存在满足给定条件的元素。

    参数：
        - pred: 要检查的条件函数，接受一个元素并返回布尔值。
        - pytree: 要检查的 pytree 对象。

    返回值：
        如果 pytree 中存在满足条件的元素，则返回 True，否则返回 False。
    """
    flat_args, _ = tree_flatten(pytree)
    return any(map(pred, flat_args))

@overload
def tree_all_only(ty: Type[T], pred: Fn[T, bool], pytree: PyTree) -> bool:
    ...

@overload
def tree_all_only(ty: Type2[T, S], pred: Fn2[T, S, bool], pytree: PyTree) -> bool:
    ...

@overload
def tree_all_only(ty: Type3[T, S, U], pred: Fn3[T, S, U, bool], pytree: PyTree) -> bool:
    ...

def tree_all_only(ty: TypeAny, pred: FnAny[bool], pytree: PyTree) -> bool:
    """检查 pytree 中特定类型的所有元素是否都满足给定条件。

    参数：
        - ty: 要检查的元素类型。
        - pred: 要检查的条件函数，接受一个元素并返回布尔值。
        - pytree: 要检查的 pytree 对象。

    返回值：
        如果 pytree 中特定类型的所有元素都满足条件，则返回 True，否则返回 False。
    """
    flat_args, _ = tree_flatten(pytree)
    return all(pred(x) for x in flat_args if isinstance(x, ty))

@overload
def tree_any_only(ty: Type[T], pred: Fn[T, bool], pytree: PyTree) -> bool:
    ...

@overload
def tree_any_only(ty: Type2[T, S], pred: Fn2[T, S, bool], pytree: PyTree) -> bool:
    ...

def tree_any_only(ty: TypeAny, pred: FnAny[bool], pytree: PyTree) -> bool:
    """检查 pytree 中特定类型的元素是否存在满足给定条件的元素。

    参数：
        - ty: 要检查的元素类型。
        - pred: 要检查的条件函数，接受一个元素并返回布尔值。
        - pytree: 要检查的 pytree 对象。

    返回值：
        如果 pytree 中特定类型的元素存在满足条件的元素，则返回 True，否则返回
    """
    flat_args, _ = tree_flatten(pytree)
    return any(pred(x) for x in flat_args if isinstance(x, ty))

# Broadcasts a pytree to the provided TreeSpec and returns the flattened
# values. If this is not possible, then this function returns None.
#
# For example, given pytree=0 and spec=TreeSpec(list, None, [LeafSpec(), LeafSpec()]),
# would return [0, 0]. This is useful for part of the vmap implementation:
# a user can pass in vmap(fn, in_dims)(*inputs). `in_dims` should be
# broadcastable to the tree structure of `inputs` and we use
# _broadcast_to_and_flatten to check this.
def _broadcast_to_and_flatten(pytree: PyTree, spec: TreeSpec) -> Optional[List[Any]]:
    """将 pytree 广播为给定的 TreeSpec 并返回展平的值。

    如果不可能进行广播，则返回 None。

    例如，给定 pytree=0 和 spec=TreeSpec(list, None, [LeafSpec(), LeafSpec()])，
    将返回 [0, 0]。这对于 vmap 实现的一部分很有用：
    用户可以通过 vmap(fn, in_dims)(*inputs) 的方式传递参数。
    `in_dims` 应该可以广播到 `inputs` 的树结构，
    我们使用 _broadcast_to_and_flatten 来进行检查。

    参数：
        - pytree: 要广播的 pytree 对象。
        - spec: 要广播到的 TreeSpec 对象。

    返回值：
        如果可以进行广播，则返回展平的值列表；否则返回 None。
    """
    assert isinstance(spec, TreeSpec)

    if _is_leaf(pytree):
        return [pytree] * spec.num_leaves
    if isinstance(spec, LeafSpec):
        return None
    node_type = _get_node_type(pytree)
    if node_type != spec.type:
        return None

    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, ctx = flatten_fn(pytree)

    # Check if the Node is different from the spec
    if len(child_pytrees) != len(spec.children_specs) or ctx != spec.context:
        return None

    # Recursively flatten the children
    result : List[Any] = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = _broadcast_to_and_flatten(child, child_spec)
        if flat is not None:
            result += flat
        else:
            return None

    return result


def pytree_to_str(spec: TreeSpec) -> str:
    """将 TreeSpec 转换为字符串表示。

    参数：
        - spec: 要转换的 TreeSpec 对象。

    返回值：
        表示 TreeSpec 的字符串。
    """
    if isinstance(spec, LeafSpec):
        return "*"
    elif spec.type in SUPPORTED_NODES:
        child_strings = [pytree_to_str(child) for child in spec.children_specs]
        return SUPPORTED_NODES[spec.type].to_str_fn(spec, child_strings)
    else:
        raise NotImplementedError(f"Serializing {spec.type} in pytree not supported yet")


def str_to_pytree(str_spec: str) -> TreeSpec:
    if str_spec == "*":
        return LeafSpec()

    for node_def in SUPPORTED_NODES.values():
        res = node_def.maybe_from_str_fn(str_spec)
        if res is not None:
            typ, context, child_strings = res
            children_spec = [
                str_to_pytree(child_string)
                for child_string in _split_nested(child_strings)
            ]
            return TreeSpec(typ, context, children_spec)
    raise NotImplementedError(f"Deserializing {str_spec} in pytree not supported yet")


def _split_nested(string: str) -> List[str]:
    """将字符串表示转换为 TreeSpec。

    参数：
        - str_spec: 要转换的字符串表示。

    返回值：
        对应的 TreeSpec 对象。
    """
    nested_parentheses = 0
    splits = []
    start_index = 0

    for i, char in enumerate(string):
        if char == "(":
            nested_parentheses += 1
        elif char == ")":
            nested_parentheses -= 1

        if nested_parentheses == 0 and char == ",":
            splits.append(string[start_index:i])
            start_index = i + 1

    splits.append(string[start_index:])
    return splits


def _parse_dict_children_spec(toplevel_str: str) -> Tuple[List[str], List[TreeSpec]]:
    """解析字典类型的子节点规范。

    参数：
        - toplevel_str: 顶层字符串表示。

    返回值：
        包含子节点上下文字符串列表和子节点规范列表的元组。
    """
    assert toplevel_str[1] == "("
    assert toplevel_str[-1] == ")"
    children_string = toplevel_str[2:-1]

    child_strings = []
    context_strings = []
    nested_parentheses = 0
    start_index = 0
    for i, char in enumerate(children_string):
        if char == ":":
            if nested_parentheses == 0:
                context_strings.append(children_string[start_index:i])
                start_index = i + 1
        elif char == "(":
            nested_parentheses += 1
        elif char == ")":
            nested_parentheses -= 1

        if nested_parentheses == 0 and char == ",":
            child_strings.append(children_string[start_index:i])
            start_index = i + 1

    child_strings.append(children_string[start_index:])
    children = [str_to_pytree(child_string) for child_string in child_strings]
    return context_strings, children
