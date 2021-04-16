import sys
from hub import Dataset
from hub.api.datasetview import DatasetView
from hub.utils import batchify_generator, batchify
from hub.compute import Transform
from typing import Iterable
from hub.exceptions import ModuleNotInstalledException
from hub.api.dataset_utils import get_value, str_to_int
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm
import hub
from hub.schema.features import featurify, SchemaDict
from hub.store.shape_detector import ShapeDetector


def get_sample_size(schema, memory_per_worker):
    """Given Schema, decides how many samples to take at once and returns it"""
    schema = featurify(schema)
    size_sum = 0
    for feature in schema._flatten():
        shp = list(feature.max_shape)
        if len(shp) == 0:
            shp = [1]

        sz = np.dtype(feature.dtype).itemsize
        if feature.dtype == "object":
            sz = (16 * 1024 * 1024 * 8) / 128

        def prod(shp):
            res = 1
            for s in shp:
                res *= s
            return res

        size = prod(shp) * sz
        size_sum += size
    samples = memory_per_worker * 1024 * 1024 / size_sum
    samples = 2 ** math.floor(math.log2(samples))
    # samples = samples * workers
    return samples


def get_temp_schema(schema, n_samples):
    schema = featurify(schema)
    new_schema = {}
    for k, v in schema.dict_.items():
        if isinstance(v, SchemaDict):
            new_schema[k] = get_temp_schema(v, n_samples)
        else:
            shapeDt = ShapeDetector(
                (10,) + v.shape,
                (10,) + v.max_shape,
                v.chunks,
                v.dtype,
                compressor=v.compressor,
            )
            if shapeDt.chunks[0] > n_samples:
                v.chunks = n_samples
                new_schema[k] = v
    return new_schema


def call_func(fn_index, item, func, func_kwargs, as_list=False):
    """Calls all the functions one after the other

    Parameters
    ----------
    fn_index: int
        The index starting from which the functions need to be called
    item:
        The item on which functions need to be applied
    as_list: bool, optional
        If true then treats the item as a list.

    Returns
    ----------
    result:
        The final output obtained after all transforms
    """
    result = item
    if fn_index < len(func):
        if as_list:
            result = [call_func(fn_index, it, func, func_kwargs) for it in result]
        else:
            result = func[fn_index](result, **func_kwargs[fn_index])
            result = call_func(
                fn_index + 1, result, func, func_kwargs, isinstance(result, list)
            )
    result = Transform._unwrap(result) if isinstance(result, list) else result
    return result


def _func_argd(item, func, func_kwargs, x=True):
    if isinstance(item, (DatasetView, Dataset)):
        item = item.numpy()
    result = call_func(
        0, item, func, func_kwargs
    )  # If the iterable obtained from iterating ds_in is a list, it is not treated as list

    if x:
        return result
    else:
        result = result if isinstance(result, list) else [result]
        # TODO replace with exception
        assert len(result) == 1 and type(result[0]) == dict
        return Transform._flatten_dict(result[0])


def upload_data(key, value, ds_out, start_idx, end_idx, value_shape):
    tensor = ds_out._tensors[f"/{key}"]
    shape = None
    if tensor.is_dynamic:
        tensor.disable_dynamicness()
        shape = tensor.get_shape_from_value([slice(start_idx, end_idx)], value)
    ds_out[
        key,
        start_idx:end_idx,
    ] = value
    value_shape[key] = (start_idx, end_idx, shape)


def empty_remote(template, **kwargs):
    """
    remote template
    """

    def wrapper(func):
        def inner(**kwargs):
            return func

        return inner

    return wrapper


try:
    import ray
    from ray.util.queue import Queue

    remote = ray.remote
except Exception:
    remote = empty_remote


class RayTransform(Transform):
    def __init__(self, func, schema, ds, scheduler="ray", workers=1, **kwargs):
        super(RayTransform, self).__init__(
            func, schema, ds, scheduler="single", workers=workers, **kwargs
        )
        self.workers = workers
        if "ray" not in sys.modules:
            raise ModuleNotInstalledException("ray")

        if not ray.is_initialized():
            ray.init(local_mode=True)

    @remote
    def transform_upload_shard(
        ds_in,
        ds_out,
        func,
        func_kwargs,
        worker_id,
        batch,
        num_workers,
        n_samples,
        schema,
    ):

        shard_size = math.ceil(len(ds_in) / num_workers)
        start_idx = worker_id * shard_size
        end_idx = min(start_idx + shard_size, len(ds_in))
        results = [
            _func_argd(ds_in[idx], func, func_kwargs, x=False)
            for idx in range(start_idx, end_idx)
        ]

        # normalize idx for ds_out
        start_idx += batch * n_samples
        end_idx += batch * n_samples
        results = Transform._split_list_to_dicts(results, schema)
        temp_dict = {}  # stores temporary data that can't be written to chunks
        value_shape = {}  # stores dynamic shape info which is written at the end

        for key, value in results.items():
            value = get_value(value)
            value = str_to_int(value, ds_out.tokenizer)
            if len(value) >= ds_out[key, 0].chunksize[0]:
                upload_data(key, value, ds_out, start_idx, end_idx, value_shape)
            else:
                temp_dict[key] = value
        return value_shape, temp_dict

    @remote
    def xtransform_upload_shard(
        ds_in,
        ds_out,
        ds_temp,
        func,
        func_kwargs,
        worker_id,
        workers,
        n_samples,
        schema,
        queue,
        pbar,
    ):
        samples = []
        value_shapes = []  # stores dynamic shape info which is written at the end
        item_count = 0
        for item in ds_in:
            item_count += 1
            results = _func_argd(item, func, func_kwargs)
            results = results if isinstance(results, list) else [results]
            samples.extend(results)
            while len(samples) >= n_samples:
                idx = queue.get()
                pbar.update(item_count)
                queue.put(idx + 1)
                print("index is", idx)
                item_count = 0
                start_idx = idx * n_samples
                end_idx = start_idx + n_samples
                current_samples = samples[:n_samples]
                current_samples = [
                    Transform._flatten_dict(item, schema=schema)
                    for item in current_samples
                ]
                current_samples = Transform._split_list_to_dicts(
                    current_samples, schema
                )
                value_shape = {}
                for key, value in current_samples.items():
                    value = get_value(value)
                    value = str_to_int(value, ds_out.tokenizer)
                    if len(value) >= ds_out[key, 0].chunksize[0]:
                        upload_data(key, value, ds_out, start_idx, end_idx, value_shape)
                    else:
                        ds_temp[
                            key,
                            start_idx:end_idx,
                        ] = value
                value_shapes.append(value_shape)
                samples = samples[n_samples:]
        ds_out.flush()
        ds_temp.flush()
        return value_shapes, samples

    def write_dynamic_shapes(self, ds_out, value_shapes_list):
        for value_shapes in value_shapes_list:
            for key, value in value_shapes.items():
                tensor = ds_out._tensors[f"/{key}"]
                if tensor.is_dynamic:
                    tensor.enable_dynamicness()
                    start_idx, end_idx, value_shape = value
                    tensor.set_dynamic_shape([slice(start_idx, end_idx)], value_shape)

    def write_temp_chunks(self, ds_out, temp_dicts, batch, n_samples):
        combined_temp_dict = defaultdict(list)
        for temp_dict in temp_dicts:
            for key, value in temp_dict.items():
                combined_temp_dict[key].extend(value)
        for key, value in combined_temp_dict.items():
            start_idx = batch * n_samples
            end_idx = start_idx + len(value)
            ds_out._tensors[f"/{key}"].enable_dynamicness()
            ds_out[
                key,
                start_idx:end_idx,
            ] = value

    def store_transform(self, ds_in, ds_out, n_samples):
        with tqdm(
            total=len(ds_in),
            unit_scale=True,
            unit=" items",
            desc=f"Storing in batches of size {n_samples}",
        ) as pbar:
            for batch, ds_in_batch in enumerate(batchify_generator(ds_in, n_samples)):
                work = [
                    self.transform_upload_shard.remote(
                        ds_in_batch,
                        ds_out,
                        self._func,
                        self.kwargs,
                        worker_id,
                        batch,
                        self.workers,
                        n_samples,
                        self.schema,
                    )
                    for worker_id in range(self.workers)
                ]
                value_shapes_list, temp_dicts = zip(*ray.get(work))
                self.write_dynamic_shapes(ds_out, value_shapes_list)
                self.write_temp_chunks(ds_out, temp_dicts, batch, n_samples)
                pbar.update(len(ds_in_batch))
        ds_out.flush()
        return ds_out

    def store_xtransform(self, ds_in, ds_out, n_samples, url, token, public):
        length = len(ds_in)
        batch_size = math.ceil(len(ds_in) / self.workers)
        temp_schema = get_temp_schema(
            self.schema, n_samples
        )  # only keeps those items whose chunks > n_samples
        ds_temp = Dataset(
            f"{url}_temp",
            shape=10 * 1000 * 1000,
            token=token,
            public=public,
            schema=temp_schema,
        )
        ds_in = batchify(ds_in, batch_size)
        queue = Queue()
        queue.put(0)

        with tqdm(
            total=length,
            unit_scale=True,
            unit=" items",
            desc="Writing large tensors to dataset",
        ) as pbar:
            work = [
                self.xtransform_upload_shard.remote(
                    ds_in[worker_id],
                    ds_out,
                    ds_temp,
                    self._func,
                    self.kwargs,
                    worker_id,
                    self.workers,
                    n_samples,
                    self.schema,
                    queue,
                    pbar,
                )
                for worker_id in range(self.workers)
            ]
            value_shapes_list, samples = zip(*ray.get(work))
            value_shapes_list = Transform._unwrap(value_shapes_list)
            # print(value_shapes_list)
            self.write_dynamic_shapes(ds_out, value_shapes_list)

            # uploading the remaining samples
            samples = Transform._unwrap(samples)
            idx = queue.get()
            start_idx = idx * n_samples
            end_idx = start_idx + len(samples)
            samples = [
                Transform._flatten_dict(item, schema=self.schema) for item in samples
            ]
            samples = Transform._split_list_to_dicts(samples, self.schema)
            for key, value in samples.items():
                ds_out[
                    key,
                    start_idx:end_idx,
                ] = value
            print(f"Size of output dataset is {end_idx}")

            # ds_out.resize_shape(end_idx + 1)
            ds_temp = ds_temp[0:start_idx]

        @hub.transform(schema=temp_schema, scheduler="ray", workers=self.workers)
        def identity(sample):
            return sample

        # writing smaller tensors to dataset
        transform_ds = identity(ds_temp)
        transform_ds.store("unused url", ds_out=ds_out, x=False)
        return ds_out

    def store(
        self,
        url: str,
        token: dict = None,
        length: int = None,
        ds: Iterable = None,
        progressbar: bool = True,
        public: bool = True,
        memory_per_worker=128,
        ds_out=None,
        x=True,
    ):
        """
        The function to apply the transformation for each element in batchified manner

        Parameters
        ----------
        url: str
            path where the data is going to be stored
        token: str or dict, optional
            If url is refering to a place where authorization is required,
            token is the parameter to pass the credentials, it can be filepath or dict
        length: int
            in case shape is None, user can provide length
        ds: Iterable
        progressbar: bool
            Show progress bar
        public: bool, optional
            only applicable if using hub storage, ignored otherwise
            setting this to False allows only the user who created it to access the dataset and
            the dataset won't be visible in the visualizer to the public

        Returns
        ----------
        ds: hub.Dataset
            uploaded dataset
        """
        _ds = ds or self.base_ds
        length = length or len(_ds)
        length = 10_000_000 if x else length
        ds_out = ds_out or self.create_dataset(
            url, length=length, token=token, public=public
        )
        n_samples = get_sample_size(self.schema, memory_per_worker)
        print(f"n_samples is {n_samples}")
        if x:
            self.store_xtransform(_ds, ds_out, n_samples, url, token, public)
        else:
            self.store_transform(_ds, ds_out, n_samples * self.workers)
        return ds_out
