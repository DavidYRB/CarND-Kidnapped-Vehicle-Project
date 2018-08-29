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
	num_particles = 20;

	default_random_engine gen;
	// sample initial state of all particles
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	for(int i=0; i<num_particles; i++){
		Particle p;
		// initialize the single particle with random sampling
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
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
	for(int i=0; i<num_particles; ++i){
		std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
		double temp_x = dist_x(gen);
		double temp_y = dist_y(gen);
		double temp_theta = dist_theta(gen);
		if(fabs(yaw_rate) > 0.0001){
			particles[i].x = temp_x + velocity/yaw_rate * (sin(temp_theta + yaw_rate*delta_t) - sin(temp_theta));
			particles[i].y = temp_y + velocity/yaw_rate * (cos(temp_theta) - cos(temp_theta + yaw_rate*delta_t));
			particles[i].theta = temp_theta + yaw_rate*delta_t;
		}
		else{
			particles[i].x += temp_x + velocity*cos(temp_theta);
			particles[i].y += temp_y + velocity*sin(temp_theta);
		}
	}
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
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
	double weights_sum = 0;
	weight_max = 0;
	std::vector<LandmarkObs> transformed_obs;
	for(int p_idx=0; p_idx < num_particles; p_idx++){
		//first, transform the observation into map coordinates 
		transformed_obs.resize(ob_map.size());
		for(int ob_idx=0; ob_idx<observations.size(); ob_idx++){
				LandmarkObs ts_ob;
				ts_ob.x = particles[p_idx].x + cos(particles[p_idx].theta)*observations[ob_idx].x - sin(particles[p_idx].theta)*observations[ob_idx].y;
				ts_ob.y = particles[p_idx].y + sin(particles[p_idx].theta)*observations[ob_idx].x + cos(particles[p_idx].theta)*observations[ob_idx].y;
				transformed_obs[ob_idx] = ts_ob;
		}
		//second, do data association (find the landmark closest to each observation and within sensor range)
		// a. find all landmarks in the sensor range make it as predicted observation
		std::vector<LandmarkObs> predicted;
		for(int lm = 0; lm < map_landmarks.landmark_list.size(); lm++){
			double dist = sqrt(pow(particles[p_idx].x - map_landmarks.landmark_list[lm].x_f, 2) \
			+ pow(particles[p_idx].y - map_landmarks.landmark_list[lm].y_f,2));
			if(dist <= sensor_range){
				LandmarkObs temp;
				temp.x = map_landmarks.landmark_list[lm].x_f;
				temp.y = map_landmarks.landmark_list[lm].y_f;
				temp.id = lm + 1;
				predicted.push_back(temp);
			}
		}
		if(predicted.size() )
		cout << "predicted size is: " << predicted.size() << endl;
		cout << "observation size is: " << transformed_obs.size() << endl;
		if(predicted.size() == 0){
			cout << "No landmarks" << endl;
		}
		// b. do data association with the real transformed observation data
		dataAssociation(predicted, transformed_obs);

		//third, update the particle with by doing normal distribution
		double w = 1.0;
		for(const auto &ob : transformed_obs){
			float landmark_x = map_landmarks.landmark_list[ob.id].x_f;
			float landmark_y = map_landmarks.landmark_list[ob.id].y_f;
			double exponent = pow((ob.x - landmark_x)/std_landmark[0], 2)/2.0\
			       				+ pow((ob.y - landmark_y)/std_landmark[1], 2)/2.0;

			w *= gauss_norm * exp(-exponent);							
		}
		particles[p_idx].weight = w;
		weights_sum += w;
	}

	// weights normalization
	
	for(int i = 0; i < particles.size(); i++){
		particles[i].weight /= weights_sum;
		if(particles[i].weight > weight_max)
			weight_max = particles[i].weight;
		weights[i] = particles[i].weight;
	}
	if(weight_max == 0){
		cout << "observition size is: " << transformed_obs.size() << endl;
		for(int i = 0; i < particles.size(); i++){
			cout << "X: " << particles[i].x << " y: " << particles[i].y\
			 << " weight: " << particles[i].weight << " " << weights[i]<< endl; 
		}
		for(int i = 0; i < transformed_obs.size(); i++){
			cout << "Ob " << i << " id: " << transformed_obs[i].id << " x: " << transformed_obs[i].x << " y: " << transformed_obs[i].y << endl;
		}

	}
	std::cout << "Current highest weight is: " << weight_max << std::endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	double beta = 0;
	std::vector<Particle> updated_p;
	updated_p.resize(num_particles);
	std::srand(static_cast <unsigned> (time(0)));
	static const double fraction = 1.0 / (RAND_MAX); 
	int index = std::rand() * fraction * num_particles;
	for(int i = 0; i < num_particles; i++){
		beta += weight_max * std::rand() * fraction;
		while(beta > weights[index]){
			beta -= weights[index++];
			index %= num_particles;
		}
		updated_p[i] = particles[index];
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
