#include "common.h"
#include <mpi.h>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <set>
#include <iostream>
#include <algorithm>
using namespace std;


int* size_to_change;
int num_bins_row, procs, extra_row_proc, proc_rows;
int* change_in_pt;
typedef vector<particle_t*> bin_t;
bin_t* bins;
vector<int> g_bid;
double size, bin_size;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void interact(particle_t* particle, int idx) {
    for (particle_t* close_particle:bins[idx]) {
        apply_force(*particle, *close_particle);
    }
}

void update_parts(particle_t curr_particle, particle_t* parts_list) {
    int part_id = curr_particle.id - 1;
    parts_list[part_id].x = curr_particle.x;
    parts_list[part_id].vx = curr_particle.vx;
    parts_list[part_id].ax = curr_particle.ax;
    parts_list[part_id].y = curr_particle.y;
    parts_list[part_id].vy = curr_particle.vy;
    parts_list[part_id].ay = curr_particle.ay;
}

int get_bid(int pos_x, int pos_y) {
    pos_x = (pos_x==num_bins_row) ? pos_x-1:pos_x;
    pos_y = (pos_y==num_bins_row) ? pos_y-1:pos_y;
    return pos_x+(pos_y*num_bins_row);
}

int get_pid(int bid) {
    int rid = bid / num_bins_row;
    int thrhld = extra_row_proc * (proc_rows + 1);
    int ret = (rid >= thrhld)?(rid - thrhld) / proc_rows + extra_row_proc:rid / (proc_rows + 1);
    return ret;
}

vector<int>* border_bids(int rank, int direction) {
    vector<int>* ret_vector = new vector<int>;
    int idx;
    if (rank < extra_row_proc) {
        idx = rank * (proc_rows + 1);
        if (direction == -1) {
            idx += proc_rows;
        }
    } else {
        idx = extra_row_proc * (proc_rows + 1) + (rank - extra_row_proc) * proc_rows;
        if (direction == -1) {
            idx += proc_rows - 1;
        }
    }
    for (int i = 0; i < num_bins_row; i++) {
        ret_vector->push_back(idx * num_bins_row + i);
    }
    return ret_vector;
}

void update_bins(particle_t* parts, int num_parts, int rank, int num_procs) {
    vector<particle_t> send_particles;
    for (int bid: g_bid) {
        auto parts = bins[bid].begin();
        while(parts != bins[bid].end()){
            particle_t& particle = **parts;
            int update = get_bid(particle.x/bin_size,particle.y/bin_size);
            if (get_pid(update) != rank) {
                send_particles.push_back(particle);
                bins[bid].erase(parts--);
            } else if(update != bid) {
                bins[update].push_back(*parts);
                bins[bid].erase(parts--);
            }
            parts+=1;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int size = send_particles.size();
    MPI_Allgather(&size, 1, MPI_INT, size_to_change, 1, MPI_INT, MPI_COMM_WORLD);
    particle_t* particles = new particle_t[num_parts];
    int total_sum = size_to_change[0];
    change_in_pt[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        change_in_pt[i] = change_in_pt[i-1] + size_to_change[i-1];
        total_sum += size_to_change[i];
    }
    MPI_Allgatherv(&send_particles[0], send_particles.size(), PARTICLE, particles, size_to_change, change_in_pt, PARTICLE, MPI_COMM_WORLD);
    for (int i = 0; i < total_sum; i++) {
        particle_t curr_particle = particles[i];
        int bin_id = get_bid(curr_particle.x/bin_size,curr_particle.y/bin_size);
        if (get_pid(bin_id) == rank) {
            bins[bin_id].push_back(&parts[curr_particle.id - 1]);
            update_parts(curr_particle, parts);
        }
    }
    delete[] particles;
}

vector<particle_t>* parts_isend(int rank, vector<MPI_Request*>* req_vector, int direction) {
    int idx = 0;
    vector<particle_t>* send_vector = new vector<particle_t>();
    MPI_Request* request = new MPI_Request();
    req_vector->push_back(request);
    vector<int>* closeby_bids = border_bids(rank, direction);
    while(idx < closeby_bids->size()) {
        int bid = closeby_bids->at(idx);
        for (auto particle : bins[bid]) {
            send_vector->push_back(*particle);
        }
        idx++;
    }
    MPI_Isend(&(*send_vector)[0], send_vector->size(), PARTICLE, rank-direction, 0, MPI_COMM_WORLD, request);
    return send_vector;
}

particle_t* parts_recv(int rank, set<int>* bins_nearby, int direction, int particle_count) {
    particle_t* buffer = new particle_t[num_bins_row+5];
    int total;
    MPI_Status status;
    MPI_Recv(buffer, particle_count, PARTICLE, rank-direction, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, PARTICLE, &total);
    for (int i = 0; i < total; i++) {
        int pos_x = buffer[i].x/bin_size;
        int pos_y = buffer[i].y/bin_size;
        int bid = get_bid(pos_x, pos_y);
        bins_nearby->insert(bid);
        bins[bid].push_back(&buffer[i]);
    }
    return buffer;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    size_to_change = (int*) malloc(num_procs * sizeof(int));
    change_in_pt = (int*) malloc(num_procs * sizeof(int));
    num_bins_row = size / cutoff;
    procs = min(num_procs, num_bins_row);
    extra_row_proc = num_bins_row % procs;
    proc_rows = num_bins_row / procs;
    bin_size = size / num_bins_row;
    bins = new bin_t[num_bins_row*num_bins_row];
    if (rank < procs) {
        int iter = (rank >= extra_row_proc) ? proc_rows : proc_rows+1;
        int row_setup = (rank >= extra_row_proc) ? extra_row_proc * (proc_rows + 1) + (rank - extra_row_proc) * proc_rows : rank * (proc_rows + 1);
        for (int i = 0; i < iter; i++) {
            for (int j = 0; j < num_bins_row; j++) {
                g_bid.push_back((row_setup+i) * num_bins_row + j);
            }
        }
        for (int i = 0; i < num_parts; i++) {
            int idx = get_bid(parts[i].x/bin_size,parts[i].y/bin_size);
            if (get_pid(idx) == rank) {
                bins[idx].push_back(&parts[i]);
            }
        }

    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    vector<MPI_Request*> requests;
    vector<vector<particle_t>*> buf_isend;
    vector<particle_t*> buf_recv;
    set<int> nearby_bins;
    bool processor_above = (rank < procs)?rank>0:false;
    bool process_below = rank < procs - 1;
    if (process_below) {
        buf_isend.push_back(parts_isend(rank, &requests, -1));
    }
    if (processor_above) {
        buf_isend.push_back(parts_isend(rank, &requests, 1));
    }
    if (processor_above) {
        buf_recv.push_back(parts_recv(rank, &nearby_bins, 1, num_parts));
    }
    if (process_below) {
        buf_recv.push_back(parts_recv(rank,&nearby_bins, -1, num_parts));
    }
    for (int bid : g_bid) { 
        for (particle_t* part : bins[bid]) {    
            part->ax = part->ay = 0;    
            bool up = bid-num_bins_row > -1;    
            bool left = bid % num_bins_row != 0;    
            bool right = bid % num_bins_row != num_bins_row - 1;    
            bool down = bid + num_bins_row < (num_bins_row * num_bins_row); 
            interact(part, bid);    
            if (right){ 
                interact(part,bid+1);   
            }   
            if (left){  
                interact(part,bid-1);   
            }   
            if (up){    
                interact(part,bid-num_bins_row);    
                if (left){  
                    interact(part,bid-num_bins_row-1);  
                }   
                if (right){ 
                    interact(part,bid-num_bins_row+1);  
                }   
            }   
            if (down){  
                interact(part,bid+num_bins_row);    
                if (left){  
                    interact(part,bid+num_bins_row-1);  
                }
                if (right){ 
                    interact(part,bid+num_bins_row+1);  
                }   
            }   
        }   
    }
    MPI_Status status_arr[requests.size()];
    for (auto bid : g_bid) {
        for (auto part : bins[bid]) {
            move(*part, size);
        }
    }

    for (int i = 0; i < buf_isend.size(); i++) {
        delete buf_isend[i];
    }
    for (auto bid : nearby_bins) {
        bins[bid].clear();
    }
    for (auto request : requests) {
        MPI_Status status;
        MPI_Wait(request, &status);
        delete request;
    }
    for (int i = 0; i < buf_recv.size(); i++) {
        delete buf_recv[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    update_bins(parts, num_parts, rank, num_procs);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    int i = 1;
    int* parts_arr = new int[num_procs];
    int* gather_arr = new int[num_procs];
    gather_arr[0] = 0;
    vector<particle_t> curr_parts;
    for (int bid: g_bid) {
        for (particle_t* particle: bins[bid]) {
            curr_parts.push_back(*particle);
        }
    }
    int temp_var = curr_parts.size();
    MPI_Gather(&temp_var, 1, MPI_INT, parts_arr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    particle_t* buf_recv = new particle_t[num_parts];
    while(i < num_procs){
        gather_arr[i] = gather_arr[i-1] + parts_arr[i-1];
        i+=1;
    }
    MPI_Gatherv(&curr_parts[0], curr_parts.size(), PARTICLE, buf_recv,parts_arr, gather_arr, PARTICLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        i = 0;
        while (i < num_parts){
            particle_t particle = buf_recv[i];
            update_parts(particle, parts);
            i+=1;
        }
    }
    delete[] parts_arr;
    delete[] gather_arr;
    delete[] buf_recv;
}