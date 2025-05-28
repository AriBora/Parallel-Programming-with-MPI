#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdbool.h>
#include "mpi.h"

#define IDX(x, y, z, nx, ny) ((z) * (ny) * (nx) + (y) * (nx) + (x))

// Function to check if the point is a local min or max at a given time step
bool is_local_extrema(float *data, int *n_ranks, float **n_values, int x, int y, int z, int nx, int ny, int nz, int nc, int t, bool find_max)
{
    /*
        data : 1D array containing the subdomain
        n_ranks : 1D array containing the ranks of neighbor in the order left, right, up, down, front, back
        n_values : 2D array containing faces from neighbors
        x, y, z : current index
        nx, ny, nz, nc : size of subdomain
        t : current time step
        find_max : bool specifying whether to find maxima (true) or minima (false)
    */
    int dx[] = {-1, 1, 0, 0, 0, 0};
    int dy[] = {0, 0, -1, 1, 0, 0};
    int dz[] = {0, 0, 0, 0, -1, 1};

    float val = data[IDX(x, y, z, nx, ny) * nc + t]; // Value at current index

    for (int i = 0; i < 6; ++i)
    {
        int nx_ = x + dx[i], ny_ = y + dy[i], nz_ = z + dz[i];

        float neighbor = 0.0;

        if (nx_ >= 0 && nx_ < nx && ny_ >= 0 && ny_ < ny && nz_ >= 0 && nz_ < nz) // If point lies inside subdomain
        {
            neighbor = data[IDX(nx_, ny_, nz_, nx, ny) * nc + t];
            if ((find_max && neighbor >= val) || (!find_max && neighbor <= val))
            {
                return false;
            }
        }

        else if (nx_ < 0 && n_ranks[0] != -1) // If point lies in the left subdomain
        {
            neighbor = n_values[0][(z * ny + y) * nc + t];
            if ((find_max && neighbor >= val) || (!find_max && neighbor <= val))
            {
                return false;
            }
        }
        else if (nx_ >= nx && n_ranks[1] != -1) // If point lies in the right subdomain
        {
            neighbor = n_values[1][(z * ny + y) * nc + t];
            if ((find_max && neighbor >= val) || (!find_max && neighbor <= val))
            {
                return false;
            }
        }
        else if (ny_ < 0 && n_ranks[2] != -1) // If point lies in the top subdomain
        {
            neighbor = n_values[2][(z * nx + x) * nc + t];
            if ((find_max && neighbor >= val) || (!find_max && neighbor <= val))
            {
                return false;
            }
        }
        else if (ny_ >= ny && n_ranks[3] != -1) // If point lies in the bottom subdomain
        {
            neighbor = n_values[3][(z * nx + x) * nc + t];
            if ((find_max && neighbor >= val) || (!find_max && neighbor <= val))
            {
                return false;
            }
        }
        else if (nz_ < 0 && n_ranks[4] != -1) // If point lies in the front subdomain
        {
            neighbor = n_values[4][(y * nx + x) * nc + t];
            if ((find_max && neighbor >= val) || (!find_max && neighbor <= val))
            {
                return false;
            }
        }
        else if (nz_ >= nz && n_ranks[5] != -1) // If point lies in the back subdomain
        {
            neighbor = n_values[5][(y * nx + x) * nc + t];
            if ((find_max && neighbor >= val) || (!find_max && neighbor <= val))
            {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    int rank, proc;

    char *file_name = argv[1];
    char *output_file_name = argv[9];

    int px = atoi(argv[2]), py = atoi(argv[3]), pz = atoi(argv[4]), nx = atoi(argv[5]), ny = atoi(argv[6]), nz = atoi(argv[7]), nc = atoi(argv[8]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double time1 = MPI_Wtime();

    // Initial Calculations
    int gsize[4] = {nz, ny, nx, nc}; // Global data size

    int xwid = nx / px;
    int ywid = ny / py;
    int zwid = nz / pz;
    int lsize[4] = {zwid, ywid, xwid, nc}; // Local subdomain size

    int x_rank = rank % px;
    int y_rank = (rank / px) % py;
    int z_rank = rank / (px * py);

    int starts[4];
    starts[0] = z_rank * zwid; // z offset
    starts[1] = y_rank * ywid; // y offset
    starts[2] = x_rank * xwid; // x offset
    starts[3] = 0;

    int *neighbour_ranks = (int *)malloc(6 * sizeof(int));
    neighbour_ranks[0] = (x_rank > 0) ? rank - 1 : MPI_PROC_NULL;            // left
    neighbour_ranks[1] = (x_rank < px - 1) ? rank + 1 : MPI_PROC_NULL;       // right
    neighbour_ranks[2] = (y_rank > 0) ? rank - px : MPI_PROC_NULL;           // up
    neighbour_ranks[3] = (y_rank < py - 1) ? rank + px : MPI_PROC_NULL;      // down
    neighbour_ranks[4] = (z_rank > 0) ? rank - px * py : MPI_PROC_NULL;      // front
    neighbour_ranks[5] = (z_rank < pz - 1) ? rank + px * py : MPI_PROC_NULL; // back

    // Domain Decomposition and File Reading
    float *buffer = (float *)malloc(xwid * ywid * zwid * nc * sizeof(float));

    MPI_Datatype filetype;
    MPI_Type_create_subarray(4, gsize, lsize, starts, MPI_ORDER_C, MPI_FLOAT, &filetype);
    MPI_Type_commit(&filetype);
    
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_seek(fh, 0, MPI_SEEK_SET);
    MPI_File_set_view(fh, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, buffer, xwid * ywid * zwid * nc, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    double time2 = MPI_Wtime();

    // Interprocess communication
    MPI_Datatype x_face, y_face;
    MPI_Type_vector(zwid, xwid * nc, xwid * ywid * nc, MPI_FLOAT, &x_face); // xz face
    MPI_Type_commit(&x_face);

    MPI_Type_vector(zwid * ywid, nc, xwid * nc, MPI_FLOAT, &y_face); // yz-face
    MPI_Type_commit(&y_face);

    float **neighbour_values = (float **)malloc(6 * sizeof(float *));
    for (int i = 0; i < 6; i++)
    {
        if (neighbour_ranks[i] != -1)
        {
            int grid_size = 0;
            if (i < 2)
            {
                grid_size = ywid * zwid * nc;
            }
            else if (i < 4)
            {
                grid_size = xwid * zwid * nc;
            }
            else
            {
                grid_size = xwid * ywid * nc;
            }
            neighbour_values[i] = (float *)malloc(grid_size * sizeof(float));
        }
        else
        {
            neighbour_values[i] = (float *)malloc(1 * sizeof(float));
        }
    }

    MPI_Request requests[6];
    MPI_Status statuses[6];
    if (neighbour_ranks[0] != -1) // If left neighbor exists
    {
        MPI_Isend(&buffer[0], 1, y_face, neighbour_ranks[0], 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Recv(neighbour_values[0], ywid * zwid * nc, MPI_FLOAT, neighbour_ranks[0], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (neighbour_ranks[1] != -1) // If right neighbor exists
    {
        MPI_Isend(&buffer[(xwid - 1) * nc], 1, y_face, neighbour_ranks[1], 1, MPI_COMM_WORLD, &requests[1]);
        MPI_Recv(neighbour_values[1], ywid * zwid * nc, MPI_FLOAT, neighbour_ranks[1], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (neighbour_ranks[2] != -1) // If top neighbor exists
    {
        MPI_Isend(&buffer[0], 1, x_face, neighbour_ranks[2], 2, MPI_COMM_WORLD, &requests[2]);
        MPI_Recv(neighbour_values[2], xwid * zwid * nc, MPI_FLOAT, neighbour_ranks[2], 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (neighbour_ranks[3] != -1) // If bottom neighbor exists
    {
        MPI_Isend(&buffer[xwid * (ywid - 1) * nc], 1, x_face, neighbour_ranks[3], 3, MPI_COMM_WORLD, &requests[3]);
        MPI_Recv(neighbour_values[3], xwid * zwid * nc, MPI_FLOAT, neighbour_ranks[3], 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (neighbour_ranks[4] != -1) // If front neighbor exists
    {
        MPI_Isend(&buffer[0], xwid * ywid * nc, MPI_FLOAT, neighbour_ranks[4], 4, MPI_COMM_WORLD, &requests[4]);
        MPI_Recv(neighbour_values[4], xwid * ywid * nc, MPI_FLOAT, neighbour_ranks[4], 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (neighbour_ranks[5] != -1) // If back neighbor exists
    {
        MPI_Isend(&buffer[xwid * ywid * (zwid - 1) * nc], xwid * ywid * nc, MPI_FLOAT, neighbour_ranks[5], 5, MPI_COMM_WORLD, &requests[5]);
        MPI_Recv(neighbour_values[5], xwid * ywid * nc, MPI_FLOAT, neighbour_ranks[5], 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Local Computation
    float *global_minima = (float *)malloc(nc * sizeof(float));
    float *global_maxima = (float *)malloc(nc * sizeof(float));
    int *local_minima = (int *)malloc(nc * sizeof(int));
    int *local_maxima = (int *)malloc(nc * sizeof(int));

    // Analyze each time step
    for (int t = 0; t < nc; ++t)
    {
        float gmin = FLT_MAX, gmax = -FLT_MAX;
        int local_min = 0, local_max = 0;

        // Loop over each index
        for (int z = 0; z < zwid; ++z)
        {
            for (int y = 0; y < ywid; ++y)
            {
                for (int x = 0; x < xwid; ++x)
                {
                    float val = buffer[IDX(x, y, z, xwid, ywid) * nc + t];

                    if (val < gmin)
                        gmin = val;
                    if (val > gmax)
                        gmax = val;

                    if (is_local_extrema(buffer, neighbour_ranks, neighbour_values, x, y, z, xwid, ywid, zwid, nc, t, false))
                    {
                        local_min++;
                    }

                    if (is_local_extrema(buffer, neighbour_ranks, neighbour_values, x, y, z, xwid, ywid, zwid, nc, t, true))
                    {
                        local_max++;
                    }
                }
            }
        }

        MPI_Reduce(&gmin, &global_minima[t], 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&gmax, &global_maxima[t], 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min, &local_minima[t], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &local_maxima[t], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    double time3 = MPI_Wtime();
    double *times = (double *)malloc(3 * sizeof(double));
    double *totaltimes = (double *)malloc(3 * sizeof(double));
    times[0] = time2 - time1; // read_time
    times[1] = time3 - time2; // main_code_time
    times[2] = time3 - time1; // total_time

    MPI_Reduce(times, totaltimes, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Output file writing
    if (rank == 0)
    {
        FILE *f = fopen(output_file_name, "w");

        for (int i = 0; i < nc - 1; i++)
        {
            fprintf(f, "(%d,%d), ", local_minima[i], local_maxima[i]);
        }
        fprintf(f, "(%d,%d)\n", local_minima[nc - 1], local_maxima[nc - 1]);

        for (int i = 0; i < nc - 1; i++)
        {
            fprintf(f, "(%.2f,%.2f), ", global_minima[i], global_maxima[i]);
        }
        fprintf(f, "(%.2f,%.2f)\n", global_minima[nc - 1], global_maxima[nc - 1]);

        fprintf(f, "%lf,%lf,%lf\n", totaltimes[0], totaltimes[1], totaltimes[2]);

        fclose(f);
    }

    for (int i = 0; i < 6; i++)
    {
        free(neighbour_values[i]);
    }
    free(neighbour_ranks);
    free(neighbour_values);
    free(buffer);

    MPI_Finalize();
    return 0;
}