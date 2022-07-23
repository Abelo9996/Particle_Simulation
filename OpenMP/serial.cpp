#include "common.h"
#include <cmath>
#include <iostream>

using namespace std;

int bin_size;
int bin_count;

class Node
{
    public:
        particle_t* particle;
        void InsertAfter(particle_t* particle);
        Node* NextNode;
};

Node *iter_node;

void Node::InsertAfter(particle_t *particle){
	Node *im_node = iter_node++;
	im_node->particle = particle;
	im_node->NextNode = this->NextNode;
	this->NextNode = im_node;
}

Node *node_list;

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

void interact(particle_t& particle, Node *node) {
    for (;node!=NULL;node=node->NextNode){
		apply_force(*node->particle, particle);
		apply_force(particle, *node->particle);
	}
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

void init_simulation(particle_t* parts, int num_parts, double size) {
	bin_size = (size/cutoff)+1;
    bin_count = bin_size*bin_size;
    node_list = new Node[bin_count+num_parts];
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    iter_node = node_list;
    int i = 0;
    int x = 0;
    int y = 0;
	for (i = 0; i < bin_count; ++i){
    	Node *node = iter_node++;
    	node->NextNode = NULL;
    	node->particle = NULL;
    }
	for (i = 0; i < num_parts; ++i) {
		x = parts[i].x/cutoff;
		y = parts[i].y/cutoff;
		node_list[x*bin_size+y].InsertAfter(&parts[i]);
		parts[i].ax = 0;
		parts[i].ay = 0;
	}
	for (i = 0; i < bin_count; ++i) {
			for (Node *start_node = node_list[i].NextNode;start_node!=NULL;start_node=start_node->NextNode) {
				if ((i/bin_size) + 1 < bin_size) {
                        if ((i%bin_size) > 0){
                            interact(*start_node->particle, node_list[i+bin_size-1].NextNode);
                        }
                        if ((i%bin_size) < bin_size+1){
                            interact(*start_node->particle, node_list[i+bin_size+1].NextNode);
                        }
                        interact(*start_node->particle, node_list[i+bin_size].NextNode);
				}
				if ((i%bin_size) + 1 < bin_size) {
					interact(*start_node->particle, node_list[i + 1].NextNode);
				}
				interact(*start_node->particle, start_node->NextNode);
			}
	}
    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}