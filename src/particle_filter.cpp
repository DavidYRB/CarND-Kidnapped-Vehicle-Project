/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// set the number of best_particles
	num_particles = 500;

	default_random_engine gen;
	// sample initial state of all particles
	std::normal_distribution<double> dist_x(0, std[0]);
	std::normal_distribution<double> dist_y(0, std[1]);
	std::normal_distribution<double> dist_theta(0, std[2]);
	for(int i=0; i<num_particles; i++){
		Particle p;
		// initialize the single particle with random sampling
		p.x = x + dist_x(gen);
		p.y = y + dist_y(gen);
		p.theta = theta + dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		// initialize the corresponding weight of the particle to 1
		weights.push_back(1);
	}

	// finish initilization
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// set noise generator
	default_random_engine gen;
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);

	for(int i=0; i<num_particles; ++i){
		if(fabs(yaw_rate) > 0.0001){
			particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
		}
		else{
			particles[i].x += velocity*cos(particles[i].theta);
			particles[i].y += velocity*sin(particles[i].theta);
		}
		particles[i].theta += yaw_rate*delta_t;

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i=0; i<observations.size(); i++){

		double closest_dist = sqrt(pow(predicted[0].x -  observations[i].x, 2) + pow(predicted[0].y -  observations[i].y, 2));
		observations[i].id = predicted[0].id;
		//std::cout << closest_dist << '\t';
		for(int j=1; j<predicted.size(); j++){
			double dist = sqrt(pow((predicted[j].x -  observations[i].x), 2) + pow(predicted[j].y -  observations[i].y, 2));
			//std::cout << dist << '\t';
			if(dist < closest_dist){
				closest_dist = dist;
 				observations[i].id = predicted[j].id;
			}
		}
		//std::cout << '\t' << closest_dist << '\n';
		//std::cout << "ob: " << observations[i].x <<'\t'<< observations[i].y <<'\t'<< observations[i].id << '\n';
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	std::vector<LandmarkObs> ob_map;
	ob_map = observations;
	double gauss_norm = 1.0/(2* M_PI * std_landmark[0] * std_landmark[1]);
	for(int p_idx=0; p_idx < num_particles; p_idx++){
		// if(p_idx == 1)
			// std::cout << "particle: " << p_idx << '\t' << particles[p_idx].x << '\t'<<particles[p_idx].y << '\n';
		std::vector<LandmarkObs> LMs_in_range;
		// find real landmarks in sensor range
		for(int lm_idx=0; lm_idx < map_landmarks.landmark_list.size(); lm_idx++){
			double dist = sqrt(pow(particles[p_idx].x - map_landmarks.landmark_list[lm_idx].x_f, 2)
			 				+ pow(particles[p_idx].y - map_landmarks.landmark_list[lm_idx].y_f,2));
			// std::cout << "dist: " << dist << '\n';
			if(dist < sensor_range){
				LandmarkObs inrange_landmark;
				inrange_landmark.x = map_landmarks.landmark_list[lm_idx].x_f;
				inrange_landmark.y = map_landmarks.landmark_list[lm_idx].y_f;
				inrange_landmark.id = lm_idx;
				// std::cout << "lm: " << inrange_landmark.x <<'\t'<< inrange_landmark.y <<'\t'<< inrange_landmark.id <<'\n';
				LMs_in_range.push_back(inrange_landmark);
				// std::cout << "LM vector size: " << LMs_in_range.size() << '\n';
			}
		}

		std::vector<LandmarkObs> transformed_obs;
		if(LMs_in_range.size()>0){
			// transformation of all observations from car coordinate to map coordinate
			for(int ob_idx=0; ob_idx<observations.size(); ob_idx++){
				LandmarkObs ts_ob;
				ts_ob.x = particles[p_idx].x + cos(particles[p_idx].theta)*observations[ob_idx].x - sin(particles[p_idx].theta)*observations[ob_idx].y;
				ts_ob.y = particles[p_idx].y + sin(particles[p_idx].theta)*observations[ob_idx].x + cos(particles[p_idx].theta)*observations[ob_idx].y;
				transformed_obs.push_back(ts_ob);
			}

			// find all landmarks for all observations
 			dataAssociation(LMs_in_range, transformed_obs);

			// std::cout << LMs_in_range.size() << '\t' << transformed_obs.size()<< '\n';

			// calculate weights as the product of all inrange observations-(closest landmarks) pair
			double w = 1.0;
			for(int ob_idx=0; ob_idx<transformed_obs.size(); ob_idx++){
				double lm_x = map_landmarks.landmark_list[transformed_obs[ob_idx].id].x_f;
				double lm_y = map_landmarks.landmark_list[transformed_obs[ob_idx].id].y_f;
				// std::cout << "LM: " << lm_x << '\t' << lm_y << '\t' << map_landmarks.landmark_list[transformed_obs[ob_idx].id].id_i<< std::endl;
				// std::cout << "Observed: " << transformed_obs[ob_idx].x << '\t' << transformed_obs[ob_idx].y << '\t' << transformed_obs[ob_idx].id << std::endl;
				double exponent = pow((transformed_obs[ob_idx].x - lm_x)/std_landmark[0], 2)/2.0
												+ pow((transformed_obs[ob_idx].y - lm_y)/std_landmark[1], 2)/2.0;

				w *= gauss_norm * exp(-exponent);
				// std::cout<< gauss_norm<< '\t' << exponent<< '\t' << w <<'\n';
			}
			particles[p_idx].weight = w;
		}
		else
			particles[p_idx].weight = 0;

		// std::cout << p_idx << ": "<<weights[p_idx] <<'\n';

	}

	// weights normalization
	double weights_sum = 0;
	for(int i=0; i<num_particles; i++){
		weights_sum += particles[i].weight;
	}
	if(weights_sum == 0)
		std::cout << "weights are zeros !!!!!!!!!!" << '\n';
	for (int i=0; i<weights.size();i++) {
		weights[i] = weights[i] / weights_sum;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	std::discrete_distribution<int> weight_dist(weights.begin(), weights.end());

	std::vector<Particle> updated_p;
	updated_p.resize(num_particles);
	// for loop select all N weights with resampling wheel algorithm
	for(int i=0; i<num_particles; i++){
		updated_p[i] = particles[weight_dist(gen)];
	}
	particles = updated_p;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

		return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
