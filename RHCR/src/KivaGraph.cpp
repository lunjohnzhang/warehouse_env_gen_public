#include "KivaGraph.h"
#include <fstream>
#include <boost/tokenizer.hpp>
#include "StateTimeAStar.h"
#include <sstream>
#include <random>
#include <chrono>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;


bool KivaGrid::get_r_mode() const
{
	return this->r_mode;
}


bool KivaGrid::get_w_mode() const
{
	return this->w_mode;
}


bool KivaGrid::load_map_from_jsonstr(std::string G_json_str)
{
	json G_json = json::parse(G_json_str);
	if (G_json["weight"])
		return load_weighted_map_from_json(G_json);
	else
		return load_unweighted_map_from_json(G_json);
}


void KivaGrid::check_mode()
{
	// r and w cannot both be true
	if (this->r_mode && this->w_mode)
	{
		throw std::invalid_argument("Map layout contains both 'r' and 'w'");
	}

	// r and w cannot both be false
	if (!this->r_mode && !this->w_mode)
	{
		throw std::invalid_argument("Map layout does not contain either 'r' or 'w'");
	}
}


/**
 * Infer simulation mode (r or w) from map layout.
*/
void KivaGrid::infer_sim_mode_from_map(json G_json)
{
	this->r_mode = false;
	this->w_mode = false;
	std::string line;
	for (int i = 0; i < this->rows; i++)
	{
		line = G_json["layout"][i];
		for (int j = 0; j < this->cols; j++)
		{
			if (line[j] == 'r')
				this->r_mode = true;
			if (line[j] == 'w')
				this->w_mode = true;
		}
	}

    check_mode();
}



bool KivaGrid::load_unweighted_map_from_json(json G_json)
{
	std::cout << "*** Loading map ***" << std::endl;
    clock_t t = std::clock();

	// Read in n_row, n_col, n_agent_loc, maxtime
	this->rows = G_json["n_row"];
	this->cols = G_json["n_col"];
	int num_endpoints, agent_num, maxtime;
	num_endpoints = G_json["n_endpoint"];
	agent_num = G_json["n_agent_loc"];
	maxtime = G_json["maxtime"];
	this->move[0] = 1;
	this->move[1] = -cols;
	this->move[2] = -1;
	this->move[3] = cols;
	this->map_name = G_json["name"];

	infer_sim_mode_from_map(G_json);

	this->types.resize(this->rows * this->cols);
	this->weights.resize(this->rows * this->cols);
	std::string line;

	for (int i = 0; i < this->rows; i++)
	{
		// getline(myfile, line);
		line = G_json["layout"][i];
		for (int j = 0; j < this->cols; j++)
		{
			int id = this->cols * i + j;
			this->weights[id].resize(5, WEIGHT_MAX);
			if (line[j] == '@') // obstacle
			{
				this->types[id] = "Obstacle";
			}
			else if (line[j] == 'e') //endpoint
			{
				this->types[id] = "Endpoint";
				this->weights[id][4] = 1;
				this->endpoints.push_back(id);
				// Under w mode, endpoints are start locations
				if (this->w_mode)
                {
                    this->agent_home_locations.push_back(id);
                }
			}
			// Only applies to r mode
			else if (line[j] == 'r' && this->r_mode) // robot rest
			{
				this->types[id] = "Home";
				this->weights[id][4] = 1;
				this->agent_home_locations.push_back(id);
			}
			// Only applies to w mode
			else if (line[j] == 'w' && this->w_mode) // workstation
			{
				this->types[id] = "Workstation";
				this->weights[id][4] = 1;
				this->workstations.push_back(id);

                // Under w mode, and with RHCR, agents can start from
                // anywhere except for obstacles.
				if (this->w_mode &&
                    !this->useDummyPaths &&
                    !this->hold_endpoints)
                {
                    this->agent_home_locations.push_back(id);
                }
			}
			else
			{
				this->types[id] = "Travel";
				this->weights[id][4] = 1;

                // Under w mode, and with RHCR, agents can start from
                // anywhere except for obstacles.
				if (this->w_mode &&
                    !this->useDummyPaths &&
                    !this->hold_endpoints)
                {
                    this->agent_home_locations.push_back(id);
                }
			}
		}
	}

	shuffle(this->agent_home_locations.begin(),this->agent_home_locations.end(), std::default_random_engine());
	for (int i = 0; i < this->cols * this->rows; i++)
	{
		if (this->types[i] == "Obstacle")
		{
			continue;
		}
		for (int dir = 0; dir < 4; dir++)
		{
			if (0 <= i + this->move[dir] && i + this->move[dir] < this->cols * this->rows && get_Manhattan_distance(i, i + this->move[dir]) <= 1 && this->types[i + this->move[dir]] != "Obstacle")
				this->weights[i][dir] = 1;
			else
				this->weights[i][dir] = WEIGHT_MAX;
		}
	}

    double runtime = (std::clock() - t) / CLOCKS_PER_SEC;
    std::cout << "Map size: " << this->rows << "x" << this->cols << " with ";

	if (this->w_mode)
	{
		std::cout << this->endpoints.size()
		<< " endpoints (home stations) and "
		<< this->workstations.size() << " workstations." << endl;
	}
	else if (this->r_mode)
	{
		std::cout << this->endpoints.size() << " endpoints and " <<
		this->agent_home_locations.size() << " home stations." << std::endl;
	}
    std::cout << "Done! (" << runtime << " s)" << std::endl;
	return true;
}

bool KivaGrid::load_weighted_map_from_json(json G_json)
{
	// TODO
	return false;
}



bool KivaGrid::load_map(std::string fname)
{
    std::size_t pos = fname.rfind('.');      // position of the file extension
    auto ext_name = fname.substr(pos, fname.size());     // get the name without extension
    if (ext_name == ".grid")
        return load_weighted_map(fname);
    else if (ext_name == ".map")
        return load_unweighted_map(fname);
    else
    {
        std::cout << "Map file name should end with either .grid or .map. " << std::endl;
        return false;
    }
}

bool KivaGrid::load_weighted_map(std::string fname)
{
	std::string line;
	std::ifstream myfile((fname).c_str());
	if (!myfile.is_open())
	{
		std::cout << "Map file " << fname << " does not exist. " << std::endl;
		return false;
	}

	std::cout << "*** Loading map ***" << std::endl;
	clock_t t = std::clock();
	std::size_t pos = fname.rfind('.');      // position of the file extension
	map_name = fname.substr(0, pos);     // get the name without extension
	getline(myfile, line); // skip the words "grid size"
	getline(myfile, line);
	boost::char_separator<char> sep(",");
	boost::tokenizer< boost::char_separator<char> > tok(line, sep);
	boost::tokenizer< boost::char_separator<char> >::iterator beg = tok.begin();
	this->rows = atoi((*beg).c_str()); // read number of cols
	beg++;
	this->cols = atoi((*beg).c_str()); // read number of rows
	move[0] = 1;
	move[1] = -cols;
	move[2] = -1;
	move[3] = cols;

	getline(myfile, line); // skip the headers

	//read tyeps and edge weights
	this->types.resize(rows * cols);
	this->weights.resize(rows * cols);
	for (int i = 0; i < rows * cols; i++)
	{
		getline(myfile, line);
		boost::tokenizer< boost::char_separator<char> > tok(line, sep);
		beg = tok.begin();
		beg++; // skip id
		this->types[i] = std::string(beg->c_str()); // read type
		beg++;
		if (types[i] == "Home")
			this->agent_home_locations.push_back(i);
		else if (types[i] == "Endpoint")
			this->endpoints.push_back(i);
		beg++; // skip x
		beg++; // skip y
		weights[i].resize(5);
		for (int j = 0; j < 5; j++) // read edge weights
		{
			if (std::string(beg->c_str()) == "inf")
				weights[i][j] = WEIGHT_MAX;
			else
				weights[i][j] = std::stod(beg->c_str());
			beg++;
		}
	}

	myfile.close();
	double runtime = (std::clock() - t) / CLOCKS_PER_SEC;
	std::cout << "Map size: " << rows << "x" << cols << " with ";
	cout << endpoints.size() << " endpoints and " <<
		agent_home_locations.size() << " home stations." << std::endl;
	std::cout << "Done! (" << runtime << " s)" << std::endl;
	return true;
}


// load map
bool KivaGrid::load_unweighted_map(std::string fname)
{
    std::string line;
    std::ifstream myfile ((fname).c_str());
	if (!myfile.is_open())
    {
	    std::cout << "Map file " << fname << " does not exist. " << std::endl;
        return false;
    }
	
    std::cout << "*** Loading map ***" << std::endl;
    clock_t t = std::clock();
	std::size_t pos = fname.rfind('.');      // position of the file extension
    map_name = fname.substr(0, pos);     // get the name without extension
    getline (myfile, line); 
	
	
	boost::char_separator<char> sep(",");
	boost::tokenizer< boost::char_separator<char> > tok(line, sep);
	boost::tokenizer< boost::char_separator<char> >::iterator beg = tok.begin();
	rows = atoi((*beg).c_str()); // read number of rows
	beg++;
	cols = atoi((*beg).c_str()); // read number of cols
	move[0] = 1;
	move[1] = -cols;
	move[2] = -1;
	move[3] = cols;

	std::stringstream ss;
	getline(myfile, line);
	ss << line;
	int num_endpoints;
	ss >> num_endpoints;

	int agent_num;
	ss.clear();
	getline(myfile, line);
	ss << line;
	ss >> agent_num;

	ss.clear();
	getline(myfile, line);
	ss << line;
	int maxtime;
	ss >> maxtime;

	//this->agents.resize(agent_num);
	//endpoints.resize(num_endpoints + agent_num);
	types.resize(rows * cols);
	weights.resize(rows*cols);
	//DeliverGoal.resize(row*col, false);
	// read map
	//int ep = 0, ag = 0;


	this->r_mode = false;
	this->w_mode = true;
	for (int i = 0; i < rows; i++)
	{
		getline(myfile, line);
		for (int j = 0; j < cols; j++)
		{
			int id = cols * i + j;
			weights[id].resize(5, WEIGHT_MAX);
			if (line[j] == '@') // obstacle
			{
				types[id] = "Obstacle";
			}
			else if (line[j] == 'e') //endpoint
			{
				types[id] = "Endpoint";
				weights[id][4] = 1;
				endpoints.push_back(id);
				// Under w mode, endpoints are start locations
				if (this->w_mode)
					this->agent_home_locations.push_back(id);
			}
			// Only applies to r mode
			else if (line[j] == 'r' && this->r_mode) // robot rest
			{
				this->types[id] = "Home";
				this->weights[id][4] = 1;
				this->agent_home_locations.push_back(id);
			}
			// Only applies to w mode
			else if (line[j] == 'w' && this->w_mode) // workstation
			{
				this->types[id] = "Workstation";
				this->weights[id][4] = 1;
				this->workstations.push_back(id);

                // Under w mode, and with RHCR, agents can start from
                // anywhere except for obstacles.
				if (this->w_mode &&
                    !this->useDummyPaths &&
                    !this->hold_endpoints)
                {
                    this->agent_home_locations.push_back(id);
                }
			}
			else
			{
				types[id] = "Travel";
				weights[id][4] = 1;

                // Under w mode, and with RHCR, agents can start from
                // anywhere except for obstacles.
				if (this->w_mode &&
                    !this->useDummyPaths &&
                    !this->hold_endpoints)
                {
                    this->agent_home_locations.push_back(id);
                }
			}
		}
	}
	shuffle(agent_home_locations.begin(), agent_home_locations.end(), std::default_random_engine());
	for (int i = 0; i < cols * rows; i++)
	{
		if (types[i] == "Obstacle")
		{
			continue;
		}
		for (int dir = 0; dir < 4; dir++)
		{
			if (0 <= i + move[dir] && i + move[dir] < cols * rows && get_Manhattan_distance(i, i + move[dir]) <= 1 && types[i + move[dir]] != "Obstacle")
				weights[i][dir] = 1;
			else
				weights[i][dir] = WEIGHT_MAX;
		}
	}
	

	myfile.close();
    double runtime = (std::clock() - t) / CLOCKS_PER_SEC;
    std::cout << "Map size: " << rows << "x" << cols << " with ";
	cout << endpoints.size() << " endpoints and " <<
	agent_home_locations.size() << " home stations." << std::endl;		
    std::cout << "Done! (" << runtime << " s)" << std::endl;
    return true;
}

void KivaGrid::preprocessing(bool consider_rotation, std::string log_dir)
{
	std::cout << "*** PreProcessing map ***" << std::endl;
	clock_t t = std::clock();
	this->consider_rotation = consider_rotation;
	fs::path table_save_path(log_dir);
	if (consider_rotation)
		table_save_path /= map_name + "_rotation_heuristics_table.txt";
	else
		table_save_path /= map_name + "_heuristics_table.txt";
	std::ifstream myfile(table_save_path.c_str());
	bool succ = false;
	if (myfile.is_open())
	{
		succ = load_heuristics_table(myfile);
		myfile.close();
	}
	if (!succ)
	{
		for (auto endpoint : this->endpoints)
		{
			this->heuristics[endpoint] = compute_heuristics(endpoint);
		}

		// Under r mode, agent home location is separated from endpoints
		if(this->r_mode)
		{
			for (auto home : this->agent_home_locations)
			{
				this->heuristics[home] = compute_heuristics(home);
			}
		}
		// Under w mode, home location is endpoints but need additional
		// heuristics to workstations
		else if(this->w_mode)
		{
			for (auto workstation : this->workstations)
			{
				this->heuristics[workstation] = compute_heuristics(workstation);
			}
		}
		cout << table_save_path << endl;
		save_heuristics_table(table_save_path.string());
	}

	double runtime = (std::clock() - t) / CLOCKS_PER_SEC;
	std::cout << "Done! (" << runtime << " s)" << std::endl;
}

double KivaGrid::get_avg_task_len(
    unordered_map<int, vector<double>> heuristics) const
{
    double total_task_len = 0.0;
    int n_tasks = 0;
    if (this->r_mode)
    {
        for (auto endpoint1 : this->endpoints)
        {
            for (auto endpoint2 : this->endpoints)
            {
                if (endpoint1 != endpoint2)
                {
                    total_task_len += heuristics[endpoint1][endpoint2];
                    n_tasks += 1;
                }
            }
        }
    }
    else if(this->w_mode)
    {
        for (auto workstation : this->workstations)
        {
            for (auto endpoint2 : this->endpoints)
            {
                total_task_len += heuristics[workstation][endpoint2];
                n_tasks += 1;
            }
        }
    }
    return total_task_len / n_tasks;
}