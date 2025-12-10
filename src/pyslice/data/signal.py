'''
Signal class and related data structures for PySlice.
Copied from sea-eco with minimal modifications.
'''

#Imports: Typing
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple, Callable, Literal, Union
from types import EllipsisType
from collections.abc import Iterable
from matplotlib.axes import Axes as mplAxes
import matplotlib.pyplot as plt
from numpy.typing import NDArray, DTypeLike

#Imports: External
from warnings import warn
from copy import deepcopy
from pathlib import Path
from inspect import signature, Parameter

from h5py import Group, File, Dataset
from uuid import uuid4

import numpy as np
import pickle

# Optional plotting imports from sea-eco
try:
    from pySEA.sea_eco._plotting.plot import plot_nd_array, PlotImage, save_plot, save_image
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plot_nd_array = None
    PlotImage = None
    save_plot = None
    save_image = None

def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid4())

def _check_and_convert_numpy(value: Any) -> Any:
    """Convert numpy scalars/arrays to native Python containers."""
    if isinstance(value, np.generic):
        return _check_and_convert_numpy(value.item())
    if isinstance(value, np.ndarray):
        return [_check_and_convert_numpy(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        converted = [_check_and_convert_numpy(v) for v in value]
        return type(value)(converted)
    return value

def safe_decode(value: Any) -> Any:
    """Safely decode bytes and numpy values to native Python objects.

    Used for HDF5 attributes (not datasets). Converts bytes to strings
    and handles special encoded values like None.
    """
    if isinstance(value, bytes):
        try:
            decoded = value.decode('utf-8')
            # Handle special encoded values
            if decoded == 'None':
                return None
            return decoded
        except UnicodeDecodeError:
            return value
    if isinstance(value, str):
        # Handle string 'None' (in case it wasn't bytes)
        if value == 'None':
            return None
    return _check_and_convert_numpy(value)

def safe_encode(value: Any) -> Any:
    """Safely encode Python values (including numpy types) for HDF5 attributes."""
    if isinstance(value, str):
        return value.encode('utf-8')
    if value is None:
        return 'None'.encode('utf-8')
    return _check_and_convert_numpy(value)

def ask_to_proceed():
    while True:
        user_input = input("Do you want to continue? (Y/N): ").upper()  # Convert to uppercase for case-insensitivity
        if user_input == 'Y' or user_input== '':
            print("Proceeding...")
            # Add the code to execute if the user chooses 'Y' here
            break  # Exit the loop once a valid 'Y' is entered
        elif user_input == 'N':
            print("Exiting...")
            # Add the code to execute if the user chooses 'N' here
            break  # Exit the loop once a valid 'N' is entered
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

def get_index_or_none(list_in:List, value:Any):
    try:
        index = list_in.index(value)
        return index
    except ValueError:
        return None

def check_dimensions_call(kwargs, fnc):
    """Check and convert dimension kwargs (dim, axis, axes) to the function's expected format then returned the allowed kwarg key.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments to check and potentially modify
    fnc : callable
        The function whose signature to check

    Returns
    -------
    str | None
        The dimension keyword that was used, or None if no dimension kwargs were found
    """
    possible_dim_keys = ['axis', 'axes', 'dim']

    # First check what dimension arguments the function accepts
    params = signature(fnc).parameters
    fnc_accepts = None
    for k in possible_dim_keys:
        if k in params:
            fnc_accepts = k
            break
    if fnc_accepts is None:
        warn('The function does not accept any dimensional references.')

    # Check which dimension arguments were provided
    provided_key = [k for k in possible_dim_keys if k in kwargs]

    if len(provided_key) == 0:
        return None
    elif len(provided_key) == 1:
        kwargs[fnc_accepts] = kwargs.pop(provided_key[0])
        return fnc_accepts
    elif len(provided_key) > 1:
        raise ValueError(f"Only one of {possible_dim_keys} can be used at a time, but got: {provided_key}")

    #kwargs[fnc_accepts] = kwargs.pop(provided_key)

    return fnc_accepts #provided_key #? should I provide the one that was found in fnc or provided

    # if prov_dim_key: kwargs[fnc_dim_key] = kwargs.pop(prov_dim_key)
    # return prov_dim_key

def get_property_dict(obj:object):
    """Get a dictionary of an objects properties

    Parameters
    ----------
    obj : Object
        Any object

    Returns
    -------
    Dict
        A dictionary of an objects properties.
    """
    cls = type(obj)
    return {
        name: getattr(obj, name)
        for name in dir(cls)
        if isinstance(getattr(cls, name), property)
    }

def get_tree_html(obj, recursive_level: int = 0,
                  exclude_keys: List[str] = [],
                  exclude_hidden: bool = True,
                  exclude_properties: bool = False,
                  promote_itterable_keys: List[str] = []
                  ) -> str:
    html = ""
    tree_dict = dict(obj.__dict__)
    if not exclude_properties: tree_dict.update(get_property_dict(obj))

    for key, value in tree_dict.items():
        #Handle exclusions and promotions of keys
        if key in exclude_keys: continue
        if exclude_hidden and key.startswith('_'): continue

        #build the html tree
        if isinstance(value, SEASerializable):
            html += (
                "<ul style='margin:0px; list-style-position:outside;'>"
                f"<details {'open' if recursive_level < 0 else ''}>"
                "<summary style='display:list-item;'>"
                f"<li style='display:inline;'><b>{key}</b></li></summary>"
                f"{value.get_tree_html(recursive_level+1)}"
                "</details></ul>"
            )
        elif isinstance(value, Iterable) and key in promote_itterable_keys:
            html += (
                        "<ul style='margin:0px; list-style-position:outside;'>"
                        f"<details {'open' if recursive_level > 0 else ''}>"
                        "<summary style='display:list-item;'>"
                        f"<li style='display:inline;'><b>{key}</b></li></summary>"
                    )
            for val in value:
                html += (
                    "<ul style='margin:0px; list-style-position:outside;'>"
                    f"<details {'open' if recursive_level > 0 else ''}>"
                    "<summary style='display:list-item;'>"
                    f"<li style='display:inline;'><b>{val}</b></li></summary>"
                    f"{val.get_tree_html(recursive_level+2)}"
                    "</details></ul>"
                )
            html += "</details></ul>"
        else:
            html += (
                "<ul style='margin:0px; list-style-position:outside;'>"
                f"<li style='margin-left:1em; padding-left:0.5em'><b>{key}</b> = {value}</li>"
                "</ul>"
            )
    return html

class SEASerializable(ABC):
    """Base class for objects that can be serialized to/from HDF5."""

    def _repr_html_(self):
        """Return a compact HTML representation of the GeneralMetadata tree."""
        return self.get_tree_html()

    def deepcopy(self):
        """Create a deep copy of the object."""
        return deepcopy(self)

    def to_dict(self,
                hidden: bool = True,
                properties: bool = False,
                exclude_keys: List[str] = [],
                deep: bool = True
                #promote_itterable_keys: List[str] = [] # Iterables atributes like Dimensions.dimensions or Signals.signals to promote. #?This was used in get_tree_html but I am not sure if it will be needed here.
                ):
        """
        Recursively convert Dimension object to a dictionary.

        Parameters
        ----------
        hidden : bool, optional
            Include hidden attributes (starting with '_'), by default Truee.
        properties : bool, optional
            Include properties, by default False.
        exclude_keys : List[str], optional
            Keys to exclude from the dictionary, by default [].
        deep : bool, optional
            Recursively convert nested SEASerializable objects, by default True.

        Returns
        -------
        dict
            Dictionary representation of the Dimension object.
        """
        to_convert = dict(self.__dict__)
        if properties: to_convert.update(get_property_dict(self))

        result = {}
        for key, value in to_convert.items():
            if not hidden and key.startswith('_'): continue #if converting hiddens then continue to the next key
            if key in exclude_keys: continue #if an exclueded key then continue to the next key

            if isinstance(value, SEASerializable):
                if deep: value = value.to_dict(hidden=hidden, properties=properties, deep=deep)
                result[key] = value
            elif isinstance(value, dict): # In case a dict is stored directly
                if deep: value = {k: v.to_dict(deep=deep) if isinstance(v, SEASerializable) else v for k, v in value.items()}
                result[key] = value
            elif isinstance(value, Tuple | List) and len(value) > 0 and isinstance(value[0], SEASerializable): #If an iterable of SEASerializable is given
                if deep: value = type(value)([v.to_dict(hidden=hidden, properties=properties, deep=deep) for v in value]) #keep the iterable type and loop it converting each element to a dict\
                result[key] = value
            else:
                result[key] = value
        return result

    def to_hdf5_group(self, parent_group: File|Group,
                force_datasets: List = [],
                name: str | None = None,
                exclude_keys: List[str] = []
                ) -> None:
        """Save object to SEA formated HDF5 group."""

        sea_type = type(self).__name__ # Get the SEA class type

        # Handle group name: use provided name, fall back to object's name attribute, or default
        if name is None:
            if hasattr(self, 'name') and getattr(self, 'name') is not None: name = getattr(self, 'name')
            else: name = sea_type

        # Create the group and assign SEA class type
        group = parent_group.create_group(name, track_order=True)
        group.attrs['sea_type'] = sea_type

        # Iteratively assign attributes by iterating to_dict.
        # Keep the to_dict local so that to_hdf5 can be called for any serializable value.
        to_write = self.to_dict(deep=False, hidden=True,
                                properties=False,
                                exclude_keys=exclude_keys)
        to_write = {**{k:v for k,v in to_write.items() if '_' not in str(k)},
                    **{k:v for k,v in to_write.items() if '_' in str(k)}}
        storage_name_counts: Dict[str, int] = {}
        for key, val in to_write.items():
            if not hasattr(self, key):
                continue
            storage_key = key[1:] if key.startswith('_') else key
            storage_name_counts[storage_key] = storage_name_counts.get(storage_key, 0) + 1
            if storage_name_counts[storage_key] > 1:
                warn(f'HDF5 serialization collision for attribute name \"{storage_key}\". Hidden and public attributes share the same name.')
            #Check the value to see if it is Serializable
            if isinstance(val, SEASerializable):
                val.to_hdf5_group(parent_group=group, name=storage_key)
                continue
            elif isinstance(val, Iterable) and not isinstance(val, str) and len(val) > 0 and all(isinstance(v, SEASerializable) for v in val):
                if key in force_datasets or storage_key in force_datasets:
                    key_group = group
                else:
                    key_group = group.create_group(name=storage_key, track_order=True)
                    key_group.attrs['sea_type'] = val.__class__.__name__ #get the iterable type as string
                for v in val:
                    v.to_hdf5_group(parent_group=key_group)
                continue

            # Check the keys to see if it should be a attribute or dataset
            if key in force_datasets or storage_key in force_datasets:
                group.create_dataset(storage_key, data=val)
            else: group.attrs[storage_key] = safe_encode(val)
        return group

    def to_sea(self, file_path: str,
                force_datasets: List = []) -> None:
        """Save object to SEA formated HDF5."""
        file_path = Path(file_path)
        if file_path.suffix != '.sea' and file_path.suffix != '':
            raise ValueError("The file extension must be '.sea' or empty.")
        file_path = str(file_path.with_suffix(''))+'.sea'

        file = File(file_path, "w")
        file.attrs['file_type'] = 'SEA-eco HDF5 file'.encode('utf-8')
        file.attrs['file_version'] = '0.0'.encode('utf-8')
        file.attrs['sea_type'] = type(self).__name__.encode('utf-8')

        self.to_hdf5_group(parent_group=file, force_datasets=force_datasets)

        file.close()

    def from_sea(self, file_path: str):
        """Load object data from an HDF5 file."""
        file_path = Path(file_path)
        if file_path.suffix != '.sea' and file_path.suffix != '':
            raise ValueError("The file extension must be '.sea' or empty.")
        file_path = str(file_path.with_suffix(''))+'.sea'
        file = File(file_path, "r")

        if len(file)!=1:
            raise ValueError("The hdf5 file contains multiple groups so can not be loaded directly. Consider using `from_hdf5_group` instead to append to the current class.")
        else:
            main_group = file[list(file.keys())[0]]

        if 'sea_type' not in main_group.attrs:
            raise ValueError("Could not locate an HDF5 group matching this object.")
        elif safe_decode(main_group.attrs['sea_type']) != type(self).__name__:
            raise ValueError(f"The HDF5 group sea_type '{safe_decode(main_group.attrs['sea_type'])}' does not match the current object type '{type(self).__name__}'.")
        else:
            self.from_hdf5_group(main_group)

        file.close()

    def from_hdf5_group(self, group: Group):
        """Populate the current object from an HDF5 group.
        """
        def _instantiate_child(sub_group: Group):
            """Instantiate an SEASerializable child using a prototype or group metadata."""
            candidate = None
            sea_type = safe_decode(sub_group.attrs.get('sea_type', b''))
            cls = globals().get(sea_type)
            if isinstance(cls, type) and issubclass(cls, SEASerializable):
                try:
                    candidate = cls()
                except TypeError:
                    candidate = cls.__new__(cls)
                    cls.__init__(candidate)
            if candidate is None: return None
            candidate.from_hdf5_group(sub_group)
            return candidate
        def _check_attr_visibility(attr: str) -> str | None:
            """_check_attr_visibility _summary_

            Parameters
            ----------
            attr : str
                Check if the attribute exists as public or private and return the correct name.

            Returns
            -------
            str | None
                _description_
            """
            if hasattr(self, f'_{attr}'): return f'_{attr}'
            else: return attr

        for key, val in group.attrs.items():
            if key == 'sea_type': continue
            attr_name = _check_attr_visibility(key)
            if attr_name is not None:
                setattr(self, attr_name, safe_decode(val))

        for key, item in group.items():
            if isinstance(item, Group):
                if 'sea_type' in item.attrs:
                    sea_type = safe_decode(item.attrs.get('sea_type', b''))
                else:
                    warn(f'Group {item.name} has no sea_type attribute. Skipping.')
                    continue
                attr_name = _check_attr_visibility(key)
                if attr_name is None:
                    warn(f'Attribute {key} from group {item.name} not found on target object.')
                    continue
                if  sea_type=='list':
                    new_items = []
                    for sub_item in item.values():
                        child = _instantiate_child(sub_item)
                        if child is not None:
                            new_items.append(child)
                    setattr(self, attr_name, new_items)
                else:
                    child = _instantiate_child(item)
                    if child is not None:
                        setattr(self, attr_name, child)
                    else:
                        warn(f'Attribute {key} could not be instantiated from group {item.name}. Skipping.')
            else:
                # This is an HDF5 dataset (array data)
                value = item[()]
                # For datasets, preserve numpy arrays - only decode bytes/strings
                if isinstance(value, bytes):
                    value = safe_decode(value)
                elif isinstance(value, np.ndarray) and value.dtype.kind in ('S', 'U', 'O'):
                    # String/object arrays - decode elements
                    value = np.array([safe_decode(v) for v in value.flat]).reshape(value.shape)
                # Otherwise keep as numpy array
                attr_name = _check_attr_visibility(key)
                if attr_name is None:
                    warn(f'Dataset {key} could not be assigned to target object.')
                    continue
                setattr(self, attr_name, value)

    def get_tree_html(self, recursive_level:int=0,
                      exclude_keys:List[str] = [],
                      exclude_hidden:bool=True,
                      exclude_properties:bool=False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
        #TODO: Have this work of of self.to_dict() instead of self.__dict__ and put the global get_tree_html() kwargs in self.to_dict
        return get_tree_html(self, recursive_level=recursive_level,
                             exclude_keys=exclude_keys,
                             exclude_hidden=exclude_hidden,
                             exclude_properties=exclude_properties,
                             promote_itterable_keys=promote_itterable_keys)

    def get_tree_str(self, pad:str='', recursive_level=None):
        """Print class as a tree.

        Parameters
        ----------
        pad : str, optional
            What to add before each entry, by default ''
        recursive_level : int, optional
           What depth to stop recursing. Not implemented, by default 0

        Returns
        -------
        _type_
            _description_
        """
        if recursive_level is not None:
            #TODO: have the tree stop at a level
            raise NotImplementedError('kwarg recursive_level not implemented yet.')
        string = ''
        N_values = len(self)
        for i, (key, value) in enumerate(self.__dict__.items()):
            if i==N_values-1: cnct = '└── '
            else: cnct = '├── '

            if isinstance(value, GeneralMetadata):
                string += f'{pad}{cnct}{key}\n'
                if i== N_values-1: pad_next = '   '
                else: pad_next = '|  '
                string += value.get_tree_str(pad=pad+pad_next)
            else:
                string += f'{pad}{cnct}{key}: {value}\n'
        return string

    def show_tree(self, show:Literal['html','str']|None='html', recursive_level=0):
        if show=='html':
            from IPython.display import display, HTML
            display(HTML(self.get_tree_html(recursive_level=recursive_level)))
        elif show=='str':
            print(self.get_tree_str(self, recursive_level=recursive_level))
        else:
            try:
                from IPython.display import display, HTML
                display(HTML(self.get_tree_html(recursive_level=recursive_level)))
            except:
                print(self.get_tree_str(self, recursive_level=recursive_level))

    def save(self, file_path: str) -> None:
        file_path = Path(file_path)
        if file_path.suffix == '.sea' or file_path.suffix == '':
            self.to_sea(file_path)
        elif file_path.suffix == '.pkl':
            with open(file_path,'wb') as f: pickle.dump(self,f)
        else:
            raise ValueError("The file extension must be '.sea', '.pkl' or empty.")

def load(file_path: str) -> SEASerializable:
    """Load a SEASerializable object from a .sea or .pkl file.

    Parameters
    ----------
    file_path : str
        Path to the file to load.

    Returns
    -------
    SEASerializable
        The loaded object.
    """
    file_path = Path(file_path)

    if file_path.suffix == '.sea' or file_path.suffix == '':
        file_path = str(file_path.with_suffix('')) + '.sea'
        with File(file_path, 'r') as f:
            if len(f) != 1:
                raise ValueError("The HDF5 file contains multiple groups. Cannot determine which to load.")
            main_group = f[list(f.keys())[0]]
            sea_type = safe_decode(main_group.attrs.get('sea_type', b''))

        # Get the class from globals and instantiate
        cls = globals().get(sea_type)
        if cls is None:
            raise ValueError(f"Unknown sea_type: {sea_type}")

        # Create instance and load data
        try:
            obj = cls()
        except TypeError:
            obj = cls.__new__(cls)
        obj.from_sea(file_path)
        return obj
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
    return obj

class GeneralMetadata(SEASerializable):
    def __init__(self, meta:dict = {}) -> None:
        """General Metadata

        Parameters
        ----------
        meta : dict
            Dictionary to be converted to metadata.
        """
        self.update_from_dict(dictionary=meta)

    def __len__(self):
        """Return the number of items in the GeneralMetadata."""
        return len(self.__dict__)

    def __repr__(self):
       """Return a compact HTML representation of the GeneralMetadata tree."""
       return self.get_tree_str()

    def to_hdf5_group(self, parent_group, force_datasets = [], name = 'Metadata'):
        return super().to_hdf5_group(parent_group, force_datasets, name)

    def update_from_dict(self, dictionary:Dict[str, Dict | List | Any | None]) -> None:
        '''Recursively update GeneralMetadata object from a dictionary.

        Parameters
        ----------
        dictionary : dict
            The key will define the Node name.
            If the value is an empty dictionary the value will be asigned as None.
            If the value is a dictionary then the value will be another GeneralMetadata object.
            Otherwise the value will be asigned directly.'''
        for key, value in dictionary.items():
            if isinstance(value, dict):
                if len(value)==0: setattr(self, key, None)
                else: setattr(self, key, GeneralMetadata(value))
            else:
                setattr(self, key, value)

    def merge(self, other: 'GeneralMetadata',
                               kind:Literal['overwrite','append','skip']='skip', warn_duplicate:bool=False,
                               inform_new=False) -> None:
        """Merge another GeneralMetadata object into this one.

        Parameters
        ----------
        other : GeneralMetadata
            The other GeneralMetadata object to merge.
        kind : bool, optional
            If True, existing keys will be overwritten. Default is False.
        warn : bool, optional
            If True, warn when duplicates arise. Default is False.
        """
        for key, value in other.__dict__.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), GeneralMetadata) and isinstance(value, GeneralMetadata):
                    getattr(self, key).merge(value, kind=kind, warn_duplicate=warn_duplicate, inform_new=inform_new)
                elif kind=='overwrite':
                    setattr(self, key, value)
                    if warn_duplicate: warn(f'Duplicate - Key {key} already exists. Overwriting.')
                elif kind=='append':
                    existing = getattr(self, key)
                    if existing != value:
                        if not isinstance(value, List): value = [value]
                        if not isinstance(existing, List): existing = [existing]
                        setattr(self, key, value+existing)
                    if warn_duplicate: warn(f'Duplicate - Key {key} already exists. Appending.')
                elif warn_duplicate: warn(f'Duplicate - Key {key} already exists. Use kind=True to overwrite.')
            else:
                setattr(self, key, value)
                if inform_new: print(f'New - Key {key} added to metadata.')

class Dimension(SEASerializable):
    """
    Dimensions of a dataset.
    """
    def __init__(self, dimension: Dict|GeneralMetadata|None = None,
                 name: str = 'Unnamed Dimension',
                 space: Literal["position", "scattering", "temporal", "spectral"] | None = None,
                 scale: float|int|Iterable[float|int]|None = None,
                 offset: float|int|Iterable[float|int]|None = None,
                 size: int|Iterable[int]|None = None,
                 units: str|Iterable[str] = '',
                 values: NDArray[Any] | Iterable[Any] = None,
                 #device: torch.device = field(default_factory=get_default_device)
                 ) -> None:
        """Initialize an Dimension object from a dictionary or GeneralMetadata object.

        Parameters
        ----------
        dimension : Dict | GeneralMetadata | None, optional
            Dict or metadata object containing, by default None
        name : str, optional
            Name of the dimension, by default 'Unnamed Dimension'
        space : Literal["position", "scattering", "temporal", "spectral"] | None, optional
           Space the dimension belongs to, by default None
        scale : float | int | Iterable[float | int] | None, optional
            Calibration scale, by default None
        offset : float | int | Iterable[float | int] | None, optional
            Calibration offset, by default None
        size : int | Iterable[int] | None, optional
            Dimension size, by default None
        units : str | Iterable[str], optional
            Calibration units, by default ''
        aixs : NDArray[Any] | Iterable[Any], optional
            Values along the dimension, by default None

        ToDo
        ----
        TODO: Add a `parametric:bool=True` and `parametric_fnc:str='x*scale+offset'` attributes
        """
        #Initialize instance attributes
        self.dimension   = dimension
        self._name   = name
        self.space  = space
        self.scale  = scale
        self.offset = offset
        self.size   = size
        self.units  = units
        self.values = values

        defaults = {'name':'Unnamed Dimension', 'space':None, 'scale':None, 'offset':None, 'units':'', 'size':None} #self.__init__.__kwdefaults__
        if self.dimension is not None and not isinstance(self._values, Iterable): #if an dimension and not values is given
            if isinstance(dimension, GeneralMetadata): self.dimension = self.dimension.to_dict() #convert to dict if GeneralMatadata
            for k,v in self.dimension.items(): #loop dimension dict and
                if k in defaults.keys() and defaults[k]==getattr(self,k): #asign the key if it is equal to the default
                    setattr(self, k, v)
        elif self.dimension is None and isinstance(self._values, Iterable): #if values and not an dimension is given
            self.size = self._values.shape[-1] #HACK: this should not be size but some sort of tuple that acounts for ndim.
        elif self.dimension is None and not isinstance(self._values, Iterable): #if no values or dimensions are supplied we set the offset and scale. We don't do it by default because these should be None if the dimension is not parametric.
            if self.offset is None: self.offset = 0
            if self.scale is None: self.scale = 1
        else:
            self._values = None
        del(self.dimension)

    def __str__(self):
        return f'{self.name}-dimension'

    def __repr__(self):
        return f'<Dimension name:{self.name} ndim:{self.ndim} size:{self.size}>'

    def __getitem__(self, key: Union[int, float, slice, tuple, EllipsisType]):
        """Support indexing and slicing of the Signal.

        Parameters
        ----------
        key : Union[int, float, slice, tuple, EllipsisType]
            Index specification. Can include integers, floats (converted to nearest index),
            slices, ellipsis, or tuples of these.
            - int: direct indexing
            - float: converted to nearest index using dimension calibration
            - slice: regular Python slicing, can include float values
            - Ellipsis: expands to cover remaining dimensions
        """
        if not isinstance(key, tuple): key = (key,)

        key_reg = tuple()
        for i, k in enumerate(key):
            if isinstance(k, float):
                k = self.find_nearest_index(k)
            elif isinstance(k, slice):
                start = k.start
                stop = k.stop
                step = k.step
                if isinstance(start, float): start = self.find_nearest_index(start)
                if isinstance(stop, float): stop = self.find_nearest_index(stop)
                if isinstance(step, float):  step = int(round(step / self.scale, 0))
                k = slice(start, stop, step)
            else:
                pass # Ellipse or int do not remove
            key_reg += (k,)

        return self.values[key_reg]

    def _check_ndim(self):
        checks = {'size':self.size,
                  'offset':self.offset,
                  'scale':self.scale}
        if all((var is not None for var in checks.values())):
            n_scale = len(self.scale) if isinstance(self.scale,Iterable) else 1
            n_offset = len(self.offset) if isinstance(self.offset,Iterable) else 1
            n_units = len(self.units) if isinstance(self.units,Iterable) and not isinstance(self.units,str) else 1
            if n_scale>1 and n_offset>1 and n_scale!=n_offset:
                raise ValueError('Scale and offset are >1 dimensions but not consistent dimensionality.')
            if n_units>2 and n_units!=max(n_scale,n_offset):
                raise ValueError('Units is >1 dimensions but not consistent dimensionality with the larger of scale or offset.')
            return max(n_scale,n_offset)
        elif self._values is not None:
            print(self._values)
            return np.ndim(self._values)
        else:
            warn('The dimensions chould not be determined from the calibrations or values.')
            return None

    @property
    def ndim(self):
        return np.ndim(self.values)
    @ndim.setter
    def ndim(self, values: Any) -> UserWarning:
        #if values is None: self._original_metadata = values
        raise UserWarning('`ndim` is read-only.')

    @property
    def values(self):
        #if self._values is None:
        checks = {'size':self.size,
                  'offset':self.offset,
                  'scale':self.scale}
        if all((var is not None for var in checks.values())):
            self._values = np.arange(self.size)*np.expand_dims(self.scale, axis=-1) + np.expand_dims(self.offset, axis=-1)
        elif self._values is not None:
            pass
        else:
            if all((var is None for var in checks.values())) and self._values is None:
                pass
            else:
                for key, var in checks.items():
                    if key not in checks.items(): warn(f'{key} is None.')
        return self._values
    @values.setter
    def values(self, values):
        self._values = values

    @property
    def name(self):
        if not isinstance(self._name, str) and isinstance(self._name, Iterable):
            return '('+', '.join(self._name)+')'
        else: return self._name
    @name.setter
    def name(self, value):
        self._name = value
        #self._check_ndim()

    def to_dict(self,
                hidden: bool = False,
                properties: bool = True,
                exclude_keys: List[str] = [],
                deep: bool = True
                ):
        """
        Recursively convert Dimension object to a dictionary.

        Parameters
        ----------
        hidden : bool, optional
            Include hidden attributes (starting with '_'), by default False.
        properties : bool, optional
            Include properties, by default True.
        exclude_keys : List[str], optional
            Keys to exclude from the dictionary, by default [].
        deep : bool, optional
            Recursively convert nested SEASerializable objects, by default True.

        Returns
        -------
        dict
            Dictionary representation of the Dimension object.
        """
        return super().to_dict(hidden=hidden, properties=properties, exclude_keys=exclude_keys, deep=deep)

    def to_hdf5_group(self, parent_group: File|Group,
                      force_datasets: List = ['_values'],
                      name: str | None = None
                      ) -> None:
        super().to_hdf5_group(parent_group=parent_group,
                              force_datasets=force_datasets,
                              name=name)
    def to_sea(self, file_path: str,
                force_datasets: List = ['_values']) -> None:
        """Save dimension to HDF5."""
        super().to_sea(file_path,
                         force_datasets=force_datasets)

    def _get_tree_html(self, recursive_level: List[str] = 0,
                       exclude_keys: List[str]= ['values'],
                       exclude_hidden: bool = True,
                       exclude_properties:bool = False,
                       promote_itterable_keys: List[str] = []
                       ) -> str:
        return super().get_tree_html(recursive_level,
                                     exclude_keys=exclude_keys,
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys
                                     )
    def get_tree_html(self, recursive_level: int = 0,
                      exclude_keys: List[str] = [],
                      exclude_hidden: bool = True,
                      exclude_properties:bool = False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
                      return self._get_tree_html(recursive_level,
                                   exclude_keys=exclude_keys+['values'],
                                   exclude_hidden=exclude_hidden,
                                   exclude_properties=exclude_properties,
                                   promote_itterable_keys=promote_itterable_keys
                                   )

    def get_calibrated_value(self, indices: int | Iterable[int]) -> float:
        """Get calibrated value at a specific index.

        Parameters
        ----------
        indices : int | Iterable[int]
            Indices in axis array.

        Returns
        -------
        float
            Value at the specified index.

        Raises
        ------
        IndexError
            Out of range index.
        """
        if self.ndim > 1:
            if len(indices) != self.ndim: raise ValueError(f"Expected {self.ndim} indices, got {len(indices)}")
            for ind, dim in zip(indices, self.size):
                if ind >= dim: raise IndexError(f"Index {ind} out of range for size {dim}")
        else:
            if indices >= self.size: raise IndexError(f"Index {indices} out of range for size {self.size}")
        return float(self.values[indices])

    def find_nearest_index(self, value: float,
                           direction:Literal['boht','above','below']='both',
                           warn_bounds=True) -> int:
        """Find the index of the nearest calibrated value.

        Parameters
        ----------
        value : float
            Calibrated value to find the nearest index to.
        direction : Literal['both','above','below']
            Direction to resolve ties. 'both' returns nearest, 'above' returns next higher index, 'below' returns next lower index. Default is 'both'.
        warn_bounds : bool, optional
            Warn if the nearest index is at the bounds of the axis, by default True.

        Returns
        -------
        int
            The nearest index.
        """
        if self.ndim > 1: raise NotImplementedError('find_nearest_index only implemented for 1D dimensions.') #TODO: Implement for multi-D dimensions.
        distances = np.abs(self.values - value)
        index = int(np.argmin(distances))
        if warn_bounds and (index==0 or index==self.size-1):
            print(f'Warning: Nearest index {index} is at the bounds of the axis (0, {self.size-1}).')
        if direction == 'both':
            return index
        elif direction == 'above':
            return index + 1
        elif direction == 'below':
            return index - 1

    def get_extent(self) -> List[float]| List[Tuple[float]]:
        """Get the extent of the dimension for plotting.

        Returns
        -------
        List[float]
            Extent as [min, max].
        """
        if self.ndim == 1:
            return [np.min(self.values[0]), np.max(self.values[-1])]
        if self.ndim > 1:
            return list(zip(np.min(self.values, axis=0), np.max(self.values, axis=0)))

class Dimensions(SEASerializable):
    def __init__(self,
                 dimensions: Iterable[Dict|Dimension] = [],
                 nav_dimensions: List[int] = [],
                 sig_dimensions: List[int] = [],
                 ) -> None:
        self.dimensions = dimensions #! This could be set as private then have __get_item__ iterate the private class.
        self.nav_dimensions = nav_dimensions
        self.sig_dimensions = sig_dimensions
        self.order = list(range(len(self.dimensions)))
        #? The below hidden dimensions are storgage for the get/set property. The getter always stores in the hidden, so is there any point in having the hiddens? I guess as is, the set would store in the hidden, then that hidden could be accessed directly allowing the user to hack if needed.
        self._spectral_dimension = []
        self._temporal_dimension = []
        self._position_dimensions = []
        self._scattering_dimensions = []

        dimensions_list = []
        for i, dimension in enumerate(self.dimensions):
            if isinstance(dimension, Dict):
                dimension_obj = Dimension(dimension)
            elif isinstance(dimension, Dimension):
                dimension_obj = dimension
            else:
                raise TypeError(f'Dimensions iterable value of {type(dimension)} was provided but is not an allowed type.')
            dimensions_list.append(dimension_obj)
        self.dimensions = dimensions_list

    @property
    def ndim(self) -> int:
        """Get the total number of dimensions across all dimensions."""
        ndim = np.sum([ax.ndim for ax in self.dimensions], dtype=int)
        return ndim
    @ndim.setter
    def ndim(self, value:Any) -> UserWarning:
        raise UserWarning('ndim should not be set by the user.')

    @property
    def spectral_dimension(self) -> int:
        for i, dimension in enumerate(self.dimensions):
            if dimension.space=='spectral':
                self._spectral_dimension = i
                break
        return self._spectral_dimension
    @spectral_dimension.setter
    def spectral_dimension(self, value:int) -> None:
        self._spectral_dimension = value

    @property
    def temporal_dimension(self) -> int:
        for i, dimension in enumerate(self.dimensions):
            if dimension.space=='temporal':
                self._temporal_dimension = i
                break
        return self._temporal_dimension
    @temporal_dimension.setter
    def temporal_dimension(self, value:int) -> None:
        self._temporal_dimension = value

    @property
    def position_dimensions(self) -> List:
        if len(self._position_dimensions)==0: # first run only, assemble based on dimension.space
            for i, dimension in enumerate(self.dimensions):
                if dimension.space=='position':
                    self._position_dimensions.append(i)
        return self._position_dimensions
    @position_dimensions.setter
    def position_dimensions(self, value:List) -> None:
        self._position_dimensions = value

    @property
    def scattering_dimensions(self) -> List:
        if len(self._scattering_dimensions)==0: # first run only, assemble based on dimension.space
            for i, dimension in enumerate(self.dimensions):
                if dimension.space=='scattering':
                    self._scattering_dimensions.append(i)
        return self._scattering_dimensions
    @scattering_dimensions.setter
    def scattering_dimensions(self, value:List) -> None:
        self._scattering_dimensions = value

    def __repr__(self):
        return f'<Dimensions ndim:{self.ndim} dimensions:[{", ".join([ax.name for ax in self.dimensions])}]>'

    def __getitem__(self, key:int|str|Iterable[int|str]):
        if isinstance(key, int): return self.dimensions[key]
        elif isinstance(key, str): return self.dimensions[self.get_index_from_name(key)]
        elif isinstance(key, Iterable):
            ret = []
            for k in key:
                if isinstance(k, int): ret.append(self.dimensions[k])
                elif isinstance(k, str): ret.append(self.dimensions[self.get_index_from_name(k)])
            return ret
        else: raise TypeError(f'Only integers and strings are allowed but a {type(key)} was provided.')

    def __len__(self):
        return len(self.dimensions)

    def add_dimension(self, dimension:Dict|Dimension) -> None:
        """Add an dimension to the Dimensions object.

        Parameters
        ----------
        dimension : Dict | Dimension
            Dictionry with dimension calibrations or Dimension class to add.

        Raises
        ------
        TypeError
            If the provided dimension is not a dictionary or Dimension object.
        """
        dimension_n = len(self.dimensions)
        if isinstance(dimension, Dict):
            dimension = Dimension(dimension)
        if isinstance(dimension, Dimension):
            self.dimensions.append(dimension)
        else: raise TypeError(f'Dimensions iterable value of {type(dimension)} was provided but is not an allowed type.')

        self.order.append(dimension_n)

    def get_names(self) -> List[str]:
        return [ax.name for ax in self.dimensions]

    def get_index_from_name(self, name:str) -> int:
        names = self.get_names()
        if name in names: return names.index(name)
        else: raise KeyError(f'A key of {name} was provided and the dimensions names are {names}')

    def get_dims_as_int(self,
                        dims: str | int | Iterable[str | int] | None
                        ) -> List | int:
        """Convert named or integer indices to integers.

        Parameters
        ----------
        dims : str | int | Iterable[str  |  int] | None
            The index to convert.

        Returns
        -------
        List | int
            List of integer indicies

        Raises
        ------
        IndexError
            _description_
        IndexError
            _description_
        """
        # Convert to int, tuple(int), or None
        if dims is None: out = None
        elif isinstance(dims, str): out = self.get_index_from_name(dims)
        elif isinstance(dims, int):
            if dims > len(self):
                raise IndexError(f'Axis index {dim} is out of bounds for signal with {self.ndim} dimensions.')
            else:
                out = dims if dims>=0 else len(self) + dims
        elif isinstance(dims, Iterable):
            out = []
            for dim in dims:
                if isinstance(dim, str):
                    out.append(self.get_index_from_name(dim))
                elif isinstance(dim, int):
                    if dim > self.ndim:
                        raise IndexError(f'Axis index {dim} is out of bounds for signal with {self.ndim} dimensions.')
                    else:
                        out.append(dim if dim>=0 else len(self.dimensions) + dim)
        return out

    def to_dict(self,
                hidden: bool = False,
                properties: bool = True,
                exclude_keys: List[str] = [],
                deep: bool = True
                ):
        """
        Recursively convert Dimension object to a dictionary.

        Parameters
        ----------
        hidden : bool, optional
            Include hidden attributes (starting with '_'), by default False.
        properties : bool, optional
            Include properties, by default True.
        exclude_keys : List[str], optional
            Keys to exclude from the dictionary, by default [].
        deep : bool, optional
            Recursively convert nested SEASerializable objects, by default True.

        Returns
        -------
        dict
            Dictionary representation of the Dimension object.
        """
        return super().to_dict(hidden=hidden, properties=properties, exclude_keys=exclude_keys, deep=deep)

    def to_hdf5_group(self, parent_group: File|Group,
                      force_datasets: List = [],
                      name: str | None = None
                      ) -> None:
        force_datasets = force_datasets#?+['dimensions']
        super().to_hdf5_group(parent_group=parent_group,
                              force_datasets=force_datasets,
                              name=name)
    def to_sea(self, file_path: str,
                force_datasets: List = []) -> None:
            """Save dimension to HDF5."""
            force_datasets = force_datasets#?+['dimensions']
            super().to_sea(file_path,
                            force_datasets=force_datasets)

    def _get_tree_html(self, recursive_level: List[str] = 0,
                       exclude_keys: List[str] = [],
                       exclude_hidden: bool = True,
                       exclude_properties:bool = False,
                       promote_itterable_keys: List[str] = ['dimensions']
                       ) -> str:
        return super().get_tree_html(recursive_level,
                                     exclude_keys=exclude_keys,
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys
                                     )
    def get_tree_html(self, recursive_level: int = 0,
                      exclude_keys: List[str] = [],
                      exclude_hidden: bool = True,
                      exclude_properties:bool = False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
                      return self._get_tree_html(recursive_level,
                                   exclude_keys=exclude_keys,
                                   exclude_hidden=exclude_hidden,
                                   exclude_properties=exclude_properties,
                                   promote_itterable_keys=promote_itterable_keys + ['dimensions']
                                   )

    def get_extents(self, kind: Literal['Axes','Image'] = 'Axes') -> List[float, List[float], List[Tuple[float]]]:
        """Get the extents of all dimensions for plotting.

        Returns
        -------
        List[float, Tuple[float]]
            List of extents as [min, max] or list of (min, max) tuples for multi-D dimensions.
        """
        extents = [dim.get_extent() for dim in self.dimensions]
        hf_sc = [dim.scale/2 for dim in self.dimensions]

        if kind=='Image' and self.ndim==2:
            # Flatten one level so extents becomes a single list like [xmin, xmax, ymax, ymin]
            extents = [extents[1][0]-hf_sc[1], extents[1][1]+hf_sc[1],
                       extents[0][1]+hf_sc[0], extents[0][0]-hf_sc[0]]
        return extents

class Signal(SEASerializable):

    def __init__(self, data: NDArray|None = None,
                 name: str = 'Signal',
                 uuid: str = None,
                 dimensions: Dimensions|Dict|None = None, #BUG Not sure if Dict will work
                 signal_type: Literal['2D-EELS','1D-EELS','Diffraction','Image']|None = None,
                 dimensions_domain:Literal['local','global'] = 'local',
                 #? metadata_domain: Literal['local','global'] = 'local',
                 original_metadata: GeneralMetadata|None = None,
                 is_lazy: bool = False,
                 metadata: GeneralMetadata|None = None
                 ):

                 self.data = data
                 self.name = name
                 self.uuid = uuid if uuid is not None else generate_uuid()
                 self.dimensions_domain = dimensions_domain
                 self.dimensions = dimensions
                 self._local_dimensions = dimensions
                 self.signal_type = signal_type
                 #self.metadata_domain = metadata_domain
                 self._original_metadata = original_metadata
                 self.is_lazy = is_lazy
                 self.metadata = metadata

                 self._parent_SignalSet: SignalSet| None = None

                #HACK This is gross. should make GeneralMetadata sliceable like dict or with ints. Also not sure this will work when not available.
                #? Might be worth thinking about where this should go. Could go in meta.instrument, even if it means all but instrument.detector is promoted, but this depends on how meta.instrument will be handled afer promotion.
                 self.detector: str = list(self.metadata.Instrument.Detectors.to_dict().keys())[0] if self.metadata is not None and hasattr(self.metadata, 'Instrument') and hasattr(self.metadata.Instrument, 'Detectors') else None

    @property
    def dimensions(self):
        if self.dimensions_domain=='global' and self._parent_SignalSet:
            names_global = self._parent_SignalSet.dimensions.get_names()
            names_local  = self._local_dimensions.get_names()
            axs_global_i = [names_global.index(nl) for nl in names_local if nl in names_global]
            # axs_local_i = [names_local.index(ng) for ng in names_global if ng in names_local]
            axs_local = [self._parent_SignalSet.dimensions.dimensions[i] for i in axs_global_i]
            nav_local = []
            sig_local = []
            order_local = []
            for i in axs_global_i:
                order_local.append(self._parent_SignalSet.dimensions.order.index(i))
                if i in self._parent_SignalSet.dimensions.nav_dimensions:
                    nav_local.append(names_local.index(names_global[i]))
                if i in self._parent_SignalSet.dimensions.sig_dimensions:
                    sig_local.append(names_local.index(names_global[i]))
            if len(nav_local)==0: nav_local = None
            if len(sig_local)==0: sig_local = None
            self._local_dimensions = Dimensions(axs_local, nav_dimensions=nav_local, sig_dimensions=sig_local)
        return self._local_dimensions
    @dimensions.setter
    def dimensions(self, dimensions):
        self._local_dimensions = dimensions
        if self.dimensions_domain=='global' and self._parent_SignalSet:
            warn('This signal is using global dimensions. Setting the the signal locally will set the calibration_domian to local.')
            while True:
                user_input = input("Do you want to continue? (Y/N): ").upper()  # Convert to uppercase for case-insensitivity
                if user_input == 'Y':
                    print("Proceeding...")
                    # Add the code to execute if the user chooses 'Y' here
                    break  # Exit the loop once a valid 'Y' is entered
                elif user_input == 'N':
                    print("Exiting...")
                    # Add the code to execute if the user chooses 'N' here
                    break  # Exit the loop once a valid 'N' is entered
                else:
                    print("Invalid input. Please enter 'Y' or 'N'.")
            self.dimensions_domain = 'local'
            self._local_dimensions = dimensions
        else:
            self._local_dimensions = dimensions

    @property
    def original_metadata(self) -> GeneralMetadata:
        return self._original_metadata
    @original_metadata.setter
    def original_metadata(self, values: Any) -> UserWarning | None:
        if self._original_metadata is None: self._original_metadata = values
        else: raise UserWarning('original_metadata is read-only. If it is necessary to change the original metadata use `_original_metadata` to set the value, but we do recomend against changing such values.')

    def __str__(self):
        return f'{self.name}-signal'

    def __repr__(self):
        return f'<Signal name="{self.name}" signal_type={self.signal_type} dimensions_domain={self.dimensions_domain}>'

    def __getitem__(self, key: Union[int, float, slice, tuple, EllipsisType]):
        """Support indexing and slicing of the Signal.

        Parameters
        ----------
        key : Union[int, float, slice, tuple, EllipsisType]
            Index specification. Can include integers, floats (converted to nearest index),
            slices, ellipsis, or tuples of these.
            - int: direct indexing
            - float: converted to nearest index using dimension calibration
            - slice: regular Python slicing, can include float values
            - Ellipsis: expands to cover remaining dimensions

        ToDo
        ----
        """
        if not isinstance(key, tuple): key = (key,)

        # Track which dimensions remain and their new sizes
        remaining_dims = self.dimensions.dimensions.copy()
        new_sizes = []

        # Handle Ellipsis expansion
        n_indices = len(key)
        n_dims = len(self.dimensions.dimensions)
        ellipsis_pos = None

        for i, k in enumerate(key):
            if k is Ellipsis:
                ellipsis_pos = i
                break

        if ellipsis_pos is not None:
            # Calculate how many dimensions the Ellipsis represents
            n_extra = n_dims - (n_indices - 1)
            # Replace Ellipsis with appropriate number of slice(None)
            expanded_key = key[:ellipsis_pos] + (slice(None),) * n_extra + key[ellipsis_pos + 1:]
        else:
            expanded_key = key

        # Ensure we don't have too many indices
        if len(expanded_key) > n_dims:
            raise IndexError(f'Too many indices: array is {n_dims}-dimensional, but {len(expanded_key)} were indexed')

        # Pad with full slices if we have too few indices
        if len(expanded_key) < n_dims:
            expanded_key = expanded_key + (slice(None),) * (n_dims - len(expanded_key))

        key_reg = tuple()
        # Now process each key with the proper dimension
        for i, k in enumerate(expanded_key):
            orig_dim_i = self.dimensions.order[i]
            orig_dim = self.dimensions.dimensions[orig_dim_i]
            if isinstance(k, int):
                remaining_dims.remove(orig_dim)
            elif isinstance(k, float):
                k = orig_dim.find_nearest_index(k)
                # This dimension is removed by indexing
                remaining_dims.remove(orig_dim)
            elif isinstance(k, slice):
                start = k.start
                stop = k.stop
                step = k.step
                if isinstance(start, float): start = orig_dim.find_nearest_index(start)
                if isinstance(stop, float): stop = orig_dim.find_nearest_index(stop)
                if isinstance(step, float):  step = int(round(step / orig_dim.scale, 0))
                k = slice(start, stop, step)
                # This dimension remains but might have a new size
                # Calculate new size for sliced dimension
                slice_indices = k.indices(orig_dim.size)
                new_size = len(range(*slice_indices))
                new_sizes.append(new_size)
            else:
                # Ellipse do not remove
                pass
            key_reg += (k,)

        sliced_data = self.data[key_reg]

        # Create new signal with updated dimensions
        new_signal = self.deepcopy()
        new_signal.data = sliced_data

        # Update dimensions for the new signal
        if remaining_dims:
            new_dims = Dimensions(
                dimensions=[dim.deepcopy() for dim in remaining_dims],
                nav_dimensions=[i for i, dim in enumerate(remaining_dims)
                              if dim in [self.dimensions[d] for d in self.dimensions.nav_dimensions]],
                sig_dimensions=[i for i, dim in enumerate(remaining_dims)
                              if dim in [self.dimensions[d] for d in self.dimensions.sig_dimensions]],
            )
            # Update sizes for sliced dimensions
            for dim, new_size in zip(new_dims.dimensions, new_sizes):
                dim.size = new_size
            new_signal.dimensions = new_dims

        return new_signal

    #TODO: Implement a wrapper for numpy functions that modifies the dimensions accordingly
    def __array__(self, dtype:DTypeLike=None) -> Any:
        """Allow numpy to treat this object as an array."""
        if dtype is None:
            return self.data
        return self.data.astype(dtype)

    def __array_function__(self, func, types, args, kwargs):
        """Handle numpy array functions like sum(), mean(), etc.

        This is called by numpy when array-like objects are passed to numpy functions.
        It differs from __array_ufunc__ which handles element-wise operations.
        """
        # Get the actual array data from any Signal objects
        arrays = []
        signal_inputs = []
        for arg in args:
            if isinstance(arg, Signal):
                arrays.append(arg.data)
                signal_inputs.append(arg)
            else:
                arrays.append(arg)

        # Convert any dimension references and get the dimension key being used
        dim_key = check_dimensions_call(kwargs, func)
        if dim_key:
            kwargs[dim_key] = self.dimensions.get_dims_as_int(kwargs[dim_key])
            if isinstance(kwargs[dim_key], List): kwargs[dim_key] = tuple(kwargs[dim_key])

            if isinstance(kwargs[dim_key], int | str | Iterable): dims_to_remove = np.atleast_1d(kwargs[dim_key])
            else: raise KeyError(f'{dim_key} takes int, str, or an Iterable and {kwargs[dim_key]} was provided.')
            remaining_dims = {i: dim for i, dim in enumerate(self.dimensions.dimensions)
                              if i not in dims_to_remove}

        # Call the numpy function with our data arrays
        result = func(*arrays, **kwargs)

        # If the result is an array, wrap it in a Signal
        if isinstance(result, np.ndarray):
            if np.ndim(result) == self.data.ndim:
                result = self.deepcopy_with_new_data(result)
            elif np.ndim(result) < self.data.ndim:
                result = self.deepcopy_with_reduced_data_dim(data=result, keep_dim=remaining_dims)
        return result

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs) -> Any:
        """Handle numpy universal functions (ufuncs).

        Parameters
        ----------
        ufunc : np.ufunc
            A numpy function that is applied to the input data.
        method : str
            How the ufunc operates on the inputs. This partially handles dimensionality expansion and contraction.
        *inputs : tuple
            Input arrays to the ufunc.
        **kwargs : dict
            Additional keyword arguments to pass to ufunc.
        """
        # Convert any Signal objects in inputs to their data arrays
        args = []
        signal_inputs = []
        for i in inputs:
            if isinstance(i, Signal):
                args.append(i.data)
                signal_inputs.append(i)
            else:
                args.append(i)

        # Check that a signal exists in the inputs
        if not signal_inputs: raise TypeError('At least one input must be a Signal object. I am not sure how this was called internally without a Signal. Please submit a detailed bug report.')

        # Catch axis indexing by name
        possible_dim_keys = ['axis', 'axes', 'dim']
        dim_key = [k for k in possible_dim_keys if k in kwargs.keys()]
        if len(dim_key) > 1:
            raise ValueError(f'Only one of {possible_dim_keys} can be used at a time, but {dim_key} were provided. Check what dimensional refference {ufunc} permits and only provide that refference.')
        elif len(dim_key) == 1:
            dim_key = dim_key[0]
        else:
            dim_key = None

        if dim_key:
            # dim is used in SEA and PyTorch but not in numpy.
            # Catch when SEA uses dim and convert to the funcitons allowed dim representation.
            if dim_key=='dim':
                for alwd_dim_key in possible_dim_keys[:-1]:
                    if alwd_dim_key in signature(ufunc).paramaters:
                        kwargs[alwd_dim_key] = kwargs['dim']
                        dim_key = alwd_dim_key
                        del kwargs['dim']
                        break

            # Convert to int, tuple(int), or None
            if kwargs[dim_key] is None: dims = None
            elif isinstance(kwargs[dim_key], str):
                dims = signal_inputs[0].dimensions.get_index_from_name(kwargs[dim_key])
            elif isinstance(kwargs[dim_key], int):
                if kwargs[dim_key] > signal_inputs[0].dimensions.ndim:
                    raise IndexError(f'Axis index {dim} is out of bounds for signal with {signal_inputs[0].dimensions.ndim} dimensions.')
                else:
                    dim = kwargs[dim_key]
                    dims = dim if dim>=0 else len(self.dimensions) + dim
            elif isinstance(kwargs[dim_key], Iterable):
                dims = tuple()
                for dim in kwargs[dim_key]:
                    if isinstance(dim, str):
                        dims += (signal_inputs[0].dimensions.get_index_from_name(dim), )
                    elif isinstance(dim, int):
                        if dim > signal_inputs[0].dimensions.ndim:
                            raise IndexError(f'Axis index {dim} is out of bounds for signal with {signal_inputs[0].dimensions.ndim} dimensions.')
                        else:
                            dim = dim if dim>=0 else len(self.dimensions) + dim
                            dims += (dim, )
            kwargs[dim_key] = dims

        # Call the ufunc on the underlying data
        result = getattr(ufunc, method)(*args, **kwargs)

        # If result is an array, wrap it back in a Signal
        if isinstance(result, np.ndarray):
            # Use the first Signal's attributes for the result
            if signal_inputs:
                base_signal = signal_inputs[0]
                out = base_signal.deepcopy()
                out.data = result

                # Update dimensions based on the ufunc operation
                if result.ndim != base_signal.data.ndim:
                    # Dimension reduction occurred
                    remaining_dims = {i: n for i, n in enumerate(base_signal.dimensions)}

                    if dim_key:
                        if isinstance(kwargs[dim_key], int): kwargs[dim_key] = tuple([kwargs[dim_key]])
                        for dim in kwargs[dim_key]: remaining_dims.pop(dim)

                    new_dims = Dimensions(
                        dimensions=remaining_dims.values(),
                        nav_dimensions=[i for i, dim in enumerate(remaining_dims)
                                      if dim in [base_signal.dimensions[d] for d in base_signal.dimensions.nav_dimensions]],
                        sig_dimensions=[i for i, dim in enumerate(remaining_dims)
                                      if dim in [base_signal.dimensions[d] for d in base_signal.dimensions.sig_dimensions]],
                    )
                    out.dimensions = new_dims

                return out

        # Return scalars or other types as-is
        return result

    def deepcopy_with_new_data(self, data:np.ndarray):
        """Copy the signal with new data."""
        out = self.deepcopy()
        out.data = data
        return out
    def deepcopy_with_reduced_data_dim(self, data:np.ndarray, keep_dim:Tuple[int|str]) -> Signal: #!
        """Copy the signal with new data of reduced dimensionality."""
        out = self.deepcopy()
        out.data = data

        # Update dimensions based on the ufunc operation
        new_dims = Dimensions(
            dimensions=[out.dimensions[i] for i in keep_dim],
            nav_dimensions=[i for i, dim in enumerate(keep_dim)
                            if dim in [self.dimensions[d] for d in self.dimensions.nav_dimensions]],
            sig_dimensions=[i for i, dim in enumerate(keep_dim)
                            if dim in [self.dimensions[d] for d in self.dimensions.sig_dimensions]],
        )
        out.dimensions = new_dims

        return out

    def to_dict(self,
                hidden: bool = False,
                properties: bool = True,
                exclude_keys: List[str] = ['data'],
                deep: bool = True
                ):
        """
        Recursively convert Dimension object to a dictionary.

        Parameters
        ----------
        hidden : bool, optional
            Include hidden attributes (starting with '_'), by default False.
        properties : bool, optional
            Include properties, by default True.
        exclude_keys : List[str], optional
            Keys to exclude from the dictionary, by default [].
        deep : bool, optional
            Recursively convert nested SEASerializable objects, by default True.

        Returns
        -------
        dict
            Dictionary representation of the Dimension object.
        """
        return super().to_dict(hidden=hidden, properties=properties, exclude_keys=exclude_keys, deep=deep)

    def to_hdf5_group(self, parent_group: File|Group,
                      force_datasets: List = ['data'],
                      name: str | None = None
                      ) -> None:
        super().to_hdf5_group(parent_group=parent_group,
                              force_datasets=force_datasets,
                              name=name)
    def to_sea(self, file_path: str,
                force_datasets: List = ['data']) -> None:
        """Save dimension to HDF5."""
        super().to_sea(file_path=file_path,
                       force_datasets=force_datasets)

    def _get_tree_html(self, recursive_level: List[str] = 0,
                       exclude_keys: List[str]= ['data', 'original_metadata'],
                       exclude_hidden: bool = True,
                       exclude_properties:bool = False,
                       promote_itterable_keys: List[str] = []
                       ) -> str:
        return super().get_tree_html(recursive_level,
                                     exclude_keys=exclude_keys,
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys
                                     )
    def get_tree_html(self, recursive_level: int = 0,
                      exclude_keys: List[str] = [],
                      exclude_hidden: bool = True,
                      exclude_properties:bool = False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
                      return self._get_tree_html(recursive_level,
                                   exclude_keys=exclude_keys+['data', 'original_metadata'],
                                   exclude_hidden=exclude_hidden,
                                   exclude_properties=exclude_properties,
                                   promote_itterable_keys=promote_itterable_keys
                                   )

    def infer_plot_dims(self,dims,fnc):
        if dims is None:
            if self.dimensions.ndim   == 2: dims = [0,1] # The signal is already plottable
            elif self.dimensions.ndim == 1: dims = [0]   # The signal is already plottable
            elif 0 < len(self.dimensions.sig_dimensions) <= 2: dims = self.dimensions.sig_dimensions # Plot the signal
            elif 0 < len(self.dimensions.nav_dimensions) <= 2: dims = self.dimensions.nav_dimensions # Plot the navigation
            else: raise ValueError('The total, signal, and navigation dimensions are larger than 2 dimensions so a dimension could not be infered. Ploting >2D is not yet implemented')
        elif dims == 'sig':
            if len(self.dimensions.sig_dimensions) <=2: dims = self.dimensions.sig_dimensions
            else: raise ValueError('The signal dimensions being plotted is larger than 2 dimensions, which is not yet implemented')
        elif dims == 'nav':
            if len(self.dimensions.nav_dimensions) <=2: dims = self.dimensions.nav_dimensions
            else: raise ValueError('The navigation dimensions being plotted is larger than 2 dimensions, which is not yet implemented')

        #TODO use self.dimensions.get_dims_as_int()
        if len(dims)>2: raise ValueError('The signal dimensions being plotted is larger than 2 dimensions, which is not yet implemented')
        else: dims = self.dimensions.get_dims_as_int(dims)

        dims_remain = tuple(i for i in range(len(self.dimensions)) if i not in dims)

        if len(dims_remain) == 0:
            return self,dims
        else:
            return fnc(self, axis=dims_remain),dims

    def show(self,
             ax: mplAxes|None = None,
             dims: None|Literal['sig','nav']|Iterable[int, str]= None,
             fnc: Callable|None = np.sum,
             filename: str|None = None,
             **kwargs
             ) -> Any:

        if not PLOTTING_AVAILABLE:
            raise ImportError("Plotting functionality requires sea-eco plotting module. Use matplotlib directly or install sea-eco.")

        sig,dims = self.infer_plot_dims(dims,fnc)

        if 'xlabel' not in kwargs:
            kwargs['xlabel'] = f'{sig.dimensions[-1].name} ({sig.dimensions[-1].units})'
        if len(dims)==2:
           if 'extent' not in kwargs: kwargs['extent'] = sig.dimensions.get_extents(kind='Image')
           if 'ylabel' not in kwargs: kwargs['ylabel'] = f'{sig.dimensions[-2].name} ({sig.dimensions[-2].units})'
           if 'scale_bar_kwargs' not in kwargs: kwargs['scale_bar_kwargs'] = {'units':sig.dimensions[-1].units}
           else:
               if 'units' not in kwargs['scale_bar_kwargs']:
                   kwargs['scale_bar_kwargs']['units'] =sig.dimensions[-1].units
        if len(dims)==1:
            if 'ylabel' not in kwargs: kwargs['xlabel'] = f'{sig.dimensions[0].name} ({sig.dimensions[0].units})'
        p = plot_nd_array(sig.data, ax=ax, **kwargs)
        if filename is not None:
            save_plot(filename)
        return p

    def image(self,
             dims: None|Literal['sig','nav']|Iterable[int, str]= None,
             fnc: Callable|None = np.sum,
             filename: str|None = None ) -> None:

        if not PLOTTING_AVAILABLE:
            raise ImportError("Plotting functionality requires sea-eco plotting module. Use matplotlib directly or install sea-eco.")

        sig,dims = self.infer_plot_dims(dims,fnc)
        size = np.asarray( [ d.get_extent() for d in sig.dimensions ] )
        size = size[:,1]-size[:,0]
        units = [ d.units for d in sig.dimensions ]
        save_image(sig.data,size,units[0],filename=filename)

class SignalSet(SEASerializable):

    def __init__(self,
                 signals: Iterable[Signal]|None = None, main_signal: int|None = None,
                 name: str|None = None,
                 metadata: GeneralMetadata|None = None,
                 dimensions: Dimensions|None = None,
                 uuid: str|None = None,
                 merge_dimensions:bool = False, merge_metadata: bool = False
                 ) -> None:

        #Initialize kwargs
        self.uuid = uuid if uuid is not None else generate_uuid()
        self.main_signal = main_signal
        self.signals = []
        self.dimensions = dimensions if dimensions is not None else Dimensions()
        self.metadata = metadata

        # Add signals
        if signals is not None: # loop signals and add them
            if main_signal is None: main_signal=0 #if main_signal is not defined define the first element as the main signal
            else: signals.insert(0, signals.pop(main_signal)) # move the main signal to the first position in the list
            self.main_signal = 0 # redefine main_signal as the first signal, as we just inforce

            for signal in signals:
                signal._parent_SignalSet = self # First set parent reference so dimensions can be accessed
                self.add_signal(signal, merge_metadata=merge_metadata, merge_dimensions=merge_dimensions) # Then add signals with metadata and dimensions merging

    def __getitem__(self, key:int|str):
        if isinstance(key, int): return self.signals[key]
        elif isinstance(key, str): return self.signals[self.get_index_from_name(key)]
        else: raise TypeError(f'Only integers and strings are allowed but a {type(key)} was provided.')

    # def to_hdf5_grop(
    # def to_sea(self, group, name):
    #     pass

    def _get_tree_html(self, recursive_level: List[str] = 0,
                       exclude_keys: List[str] = [],
                       exclude_hidden: bool = True,
                       exclude_properties:bool = False,
                       promote_itterable_keys: List[str] = ['signals']
                       ) -> str:
        return super().get_tree_html(recursive_level,
                                     exclude_keys=exclude_keys,
                                     exclude_hidden=exclude_hidden,
                                     exclude_properties=exclude_properties,
                                     promote_itterable_keys=promote_itterable_keys
                                     )
    def get_tree_html(self, recursive_level: int = 0,
                      exclude_keys: List[str] = [],
                      exclude_hidden: bool = True,
                      exclude_properties:bool = False,
                      promote_itterable_keys: List[str] = []
                      ) -> str:
                      return self._get_tree_html(recursive_level,
                                   exclude_keys=exclude_keys,
                                   exclude_hidden=exclude_hidden,
                                   exclude_properties=exclude_properties,
                                   promote_itterable_keys=promote_itterable_keys + ['signals']
                                   )

    def add_signal(self, signal,
                   merge_metadata:bool=False,
                   merge_dimensions:bool=False,
                   meta_kwargs=dict(kind='append', warn_duplicate=False, inform_new=False)):
        signal = signal.deepcopy() #? Should this be coppied?
        if merge_metadata:
            if self.metadata is None: self.metadata = signal.metadata
            elif signal.metadata is not None: self.metadata.merge(signal.metadata, **meta_kwargs)

        if merge_dimensions:
            # dimensions is initialized in __init__, so this case shouldn't happen anymore
            if len(self.dimensions.nav_dimensions)==0 and len(signal.dimensions.nav_dimensions)!=0:
                self.dimensions.nav_dimensions = signal.dimensions.nav_dimensions
                if len(self.dimensions.sig_dimensions)==0 and len(signal.dimensions.sig_dimensions)!=0:
                    self.dimensions.sig_dimensions = signal.dimensions.sig_dimensions
                if len(self.dimensions.nav_dimensions)==0 and len(signal.dimensions.nav_dimensions)!=0:
                    self.dimensions.nav_dimensions = signal.dimensions.nav_dimensions
                for ax in signal.dimensions.dimensions:
                    if ax.name not in self.dimensions.get_names():
                        self.dimensions.add_dimension(ax)
            signal.dimensions_domain = 'global'
        self.signals.append(signal)

    def get_names(self) -> List[str]:
        return [sig.name for sig in self.signals]

    def get_index_from_name(self, name:str) -> int:
        names = self.get_names()
        if name in names: return names.index(name)
        else: raise KeyError(f'A key of {name} was provided and the signal names are {names}')

class AcquisitionSet(SignalSet):

    def __init__(self,
                 signals: Iterable[Signal]|None = None, main_signal: int|None = None,
                 metadata: GeneralMetadata|None = None,
                 dimensions: Dimensions|None = None,
                 merge_dimensions: bool=True, merge_metadata: bool = True,
                 instrument_uuid: str|None = None
                 ) -> None:

        super().__init__(signals=signals, main_signal=main_signal,
                         metadata=metadata,
                         dimensions=dimensions,
                         merge_dimensions=merge_dimensions, merge_metadata=merge_metadata)
        if self._check_aquisition_uuids():
            self.uuid = self.signals[0].metadata.Instrument.Scan.scan_uuid
        else:
            self.uuid = generate_uuid()
        self.instrument_uuid = instrument_uuid

    def _check_aquisition_uuids(self):
        uuids = [sig.metadata.Instrument.Scan.scan_uuid for sig in self.signals]
        if len(self.signals)==0:
            return False
        elif len(set(uuids))!=1:
            warn('Not all signals in this set have the same scan uuid.')
            return False
        return True
