import xarray as xr
import h5py



def task_to_dataarray(task):
    return xr.DataArray(data=task.value, coords=[(d.label, d.values()[0].value) for d in task.dims])

def dedalus_to_xarray(filename):
    """Convert dedalus output into a xarray format.

    Dedalus outputs HDF5 files with two sections '"""
    with h5py.File(filename, mode='r') as f:
        tasks = {}
        for task_name in f['tasks']:
            xarr = task_to_dataarray(f['tasks'][task_name])
            tasks[task_name] =  xarr
        dset = xr.Dataset(tasks).rename({'t': 'time'})

        # write all the different time coordinates provided
        for tscale in ['sim_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
            dset.coords[tscale] = ('time', f['scales'][tscale].value)

    return dset

if __name__ == '__main__':
    import sys
    try:
        input = sys.argv[1]
        output= sys.argv[2]
    except:
        print("Usage:  data_util.py input_file.h5 output_file.nc")
    dset = dedalus_to_xarray(input)
    dset.to_netcdf(output)
