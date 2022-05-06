#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "MeanShiftSegmentation.hpp"
#include <vector>
#include <stdexcept>

/*
	This implementation of mean shift segmentation was extracted whole cloth
	from the EDISON codebase made freely available by RIUL, the Robust Image
	Understanding Library of Rutgers University (see here
	http://coewww.rutgers.edu/riul/research/code/EDISON/index.html)
	It's original authors were Bogdan Georgescu and Chris M. Christoudias. 

	Georgescu and Christoudias's C code has been ported herein to a thin C++ 
	wrapper with minor changes made to remove some C-isms and ported to use 
	OpenCV types and generally re-factored for OpenCV compatability, 
	by Joe Wezorek, reachable at http://jwezorek.com. 
	
	The most significant change coming out of this re-factoring work in terms of 
	functionality and/or performance was replacing the EDISON's codebase's 
	L*u*v*-to-RGB/RGB-to-L*u*v* conversion routines with OpenCV calls. This actually 
	changes the output of this code relative to EDISON because OpenCV and 
	EDISON give different Luv values for the same image. Not sure who is right or
	the meaning of the difference but OpenCV is an industry standard so am erring 
	on the side of OpenCV and further the segmentation this code outputs is in 
	my opinion better that what results from EDISON's Luv routines while 
	performance is unchanged.

	If anyone is interested in contributing to this codebase the next logical thing
	to would be 
		(1) replace the EDISON's Fill(...) routine with cv::floodFill
		(2) replace the C style implementation of the region adjancency table
		    with idiomatic C++ i.e. use a std::vector<std::list> instead of
			ad hoc linked lists.
	-Joe
*/

namespace {
	const int KP = 2;
	const double EPSILON = 0.01;
	const double TC_DIST_FACTOR = 0.5;
	const int LIMIT = 100;
	const float LUV_THRESHOLD = 0.1f;
	const int NODE_MULTIPLE = 10;

	cv::Mat RgbToLuv(const cv::Mat& img)
	{
		cv::Mat float_img;
		img.convertTo(float_img, (img.channels() == 3) ? CV_32FC3 : CV_32FC1);

		float_img *= 1.0f / 255.0f;
		cv::Mat luv;
		cv::cvtColor(float_img, luv, cv::COLOR_BGR2Luv);

		return luv;
	}

	cv::Mat LuvToRgb(const cv::Mat& img)
	{
		cv::Mat rgb_float;
		cv::cvtColor(img, rgb_float, cv::COLOR_Luv2BGR);
		rgb_float *= 255.0f;

		cv::Mat output;
		rgb_float.convertTo(output, (img.channels() == 3) ? CV_8UC3 : CV_8UC1);

		return output;
	}

	class MeanShiftSegmentationImpl : public MeanShiftSegmentation {
	private:
		int connectivity_;
		float sigma_r_;
		int sigma_s_;
		bool optimized_;
		int min_size_;
		float speed_threshold_;

		class RgnAdjList {
		public:

			int		label;
			float	edge_strength;
			int		edge_pixel_count;

			RgnAdjList	*next;

			RgnAdjList(void) {
				label = -1;
				next = NULL;
				edge_strength = 0;
				edge_pixel_count = 0;
			}

			~RgnAdjList(void) {
			}

			int Insert(RgnAdjList* entry) {

				//if the list contains only one element
				//then insert this element into next
				if (!next)
				{
					//insert entry
					next = entry;
					entry->next = NULL;

					//done
					return 0;
				}

				//traverse the list until either:

				//(a) entry's label already exists - do nothing
				//(b) the list ends or the current label is
				//    greater than entry's label, thus insert the entry
				//    at this location

				//check first entry
				if (next->label > entry->label)
				{
					//insert entry into the list at this location
					entry->next = next;
					next = entry;

					//done
					return 0;
				}

				//check the rest of the list...
				exists = 0;
				cur = next;
				while (cur)
				{
					if (entry->label == cur->label)
					{
						//node already exists
						exists = 1;
						break;
					}
					else if ((!(cur->next)) || (cur->next->label > entry->label))
					{
						//insert entry into the list at this location
						entry->next = cur->next;
						cur->next = entry;
						break;
					}

					//traverse the region adjacency list
					cur = cur->next;
				}

				//done. Return exists indicating whether or not a new node was
				//      actually inserted into the region adjacency list.
				return (int)(exists);
			}

		private:
			RgnAdjList	*cur, *prev;
			unsigned char exists;
		};

		struct MeanShiftSegmentationState
		{
			cv::Mat image;
			std::vector<float> modes;
			cv::Mat labels;
			std::vector<int> mode_point_counts;
			std::vector<int> index_table;
			std::vector<int> mode_table;
			std::vector<int> point_list;
			int point_count;
			float h[2];
			int rgn_count;

			MeanShiftSegmentationState(const cv::Mat& inp, cv::OutputArray seg, cv::OutputArray lbls) :
				h{ 1.0f,1.0f } {
				int L = inp.rows * inp.cols;
				int N = inp.channels();

				image = cv::Mat(inp.rows, inp.cols, inp.type());

				//lbls.create(inp.rows, inp.cols, (N == 3) ? CV_32SC3 : CV_32SC1);
				lbls.create(inp.rows, inp.cols, CV_32SC1);
				labels = lbls.getMat();

				modes.resize(L*(N + 2));
				mode_point_counts.resize(L);
				index_table.resize(L);
				mode_table.resize(L);
				point_list.resize(L);
				point_count = 0;
				rgn_count = 0;
			}
		};

		struct RegionAdjacencyTable
		{
			std::vector<RgnAdjList> rgn_adj_list;
			std::vector<RgnAdjList> rgn_adj_pool;
			RgnAdjList* free_rgn_adj_lists;

			RegionAdjacencyTable(const MeanShiftSegmentationState& state)
			{
				int width = state.image.cols;
				int height = state.image.rows;
				const int* labels = state.labels.ptr<int>();

				//Allocate memory for region adjacency matrix if it hasn't already been allocated
				rgn_adj_list.resize(state.rgn_count);
				rgn_adj_pool.resize(NODE_MULTIPLE * state.rgn_count);

				//initialize the region adjacency list
				int i;
				for (i = 0; i < state.rgn_count; i++)
				{
					rgn_adj_list[i].edge_strength = 0;
					rgn_adj_list[i].edge_pixel_count = 0;
					rgn_adj_list[i].label = i;
					rgn_adj_list[i].next = NULL;
				}

				//initialize RAM free list
				free_rgn_adj_lists = &rgn_adj_pool[0];
				for (i = 0; i < NODE_MULTIPLE * state.rgn_count - 1; i++)
				{
					rgn_adj_pool[i].edge_strength = 0;
					rgn_adj_pool[i].edge_pixel_count = 0;
					rgn_adj_pool[i].next = &rgn_adj_pool[i + 1];
				}
				rgn_adj_pool[NODE_MULTIPLE*state.rgn_count - 1].next = NULL;

				//traverse the labeled image building
				//the RAM by looking to the right of
				//and below the current pixel location thus
				//determining if a given region is adjacent
				//to another
				int		j, curLabel, rightLabel, bottomLabel, exists;
				RgnAdjList	*raNode1, *raNode2, *oldRAFreeList;
				for (i = 0; i < height - 1; i++)
				{
					//check the right and below neighbors
					//for pixel locations whose x < width - 1
					for (j = 0; j < width - 1; j++)
					{
						//calculate pixel labels
						curLabel = labels[i*width + j];	//current pixel
						rightLabel = labels[i*width + j + 1];	//right   pixel
						bottomLabel = labels[(i + 1)*width + j];	//bottom  pixel

																	//check to the right, if the label of
																	//the right pixel is not the same as that
																	//of the current one then region[j] and region[j+1]
																	//are adjacent to one another - update the RAM
						if (curLabel != rightLabel)
						{
							//obtain RAList object from region adjacency free
							//list
							raNode1 = free_rgn_adj_lists;
							raNode2 = free_rgn_adj_lists->next;

							//keep a pointer to the old region adj. free
							//list just in case nodes already exist in respective
							//region lists
							oldRAFreeList = free_rgn_adj_lists;

							//update region adjacency free list
							free_rgn_adj_lists = free_rgn_adj_lists->next->next;

							//populate RAList nodes
							raNode1->label = curLabel;
							raNode2->label = rightLabel;

							//insert nodes into the RAM
							exists = 0;
							rgn_adj_list[curLabel].Insert(raNode2);
							exists = rgn_adj_list[rightLabel].Insert(raNode1);

							//if the node already exists then place
							//nodes back onto the region adjacency
							//free list
							if (exists)
								free_rgn_adj_lists = oldRAFreeList;

						}

						//check below, if the label of
						//the bottom pixel is not the same as that
						//of the current one then region[j] and region[j+width]
						//are adjacent to one another - update the RAM
						if (curLabel != bottomLabel)
						{
							//obtain RAList object from region adjacency free
							//list
							raNode1 = free_rgn_adj_lists;
							raNode2 = free_rgn_adj_lists->next;

							//keep a pointer to the old region adj. free
							//list just in case nodes already exist in respective
							//region lists
							oldRAFreeList = free_rgn_adj_lists;

							//update region adjacency free list
							free_rgn_adj_lists = free_rgn_adj_lists->next->next;

							//populate RAList nodes
							raNode1->label = curLabel;
							raNode2->label = bottomLabel;

							//insert nodes into the RAM
							exists = 0;
							rgn_adj_list[curLabel].Insert(raNode2);
							exists = rgn_adj_list[bottomLabel].Insert(raNode1);

							//if the node already exists then place
							//nodes back onto the region adjacency
							//free list
							if (exists)
								free_rgn_adj_lists = oldRAFreeList;

						}

					}

					//check only to the bottom neighbors of the right boundary
					//pixels...

					//calculate pixel locations (j = width-1)
					curLabel = labels[i*width + j];	//current pixel
					bottomLabel = labels[(i + 1)*width + j];	//bottom  pixel

																//check below, if the label of
																//the bottom pixel is not the same as that
																//of the current one then region[j] and region[j+width]
																//are adjacent to one another - update the RAM
					if (curLabel != bottomLabel)
					{
						//obtain RAList object from region adjacency free
						//list
						raNode1 = free_rgn_adj_lists;
						raNode2 = free_rgn_adj_lists->next;

						//keep a pointer to the old region adj. free
						//list just in case nodes already exist in respective
						//region lists
						oldRAFreeList = free_rgn_adj_lists;

						//update region adjacency free list
						free_rgn_adj_lists = free_rgn_adj_lists->next->next;

						//populate RAList nodes
						raNode1->label = curLabel;
						raNode2->label = bottomLabel;

						//insert nodes into the RAM
						exists = 0;
						rgn_adj_list[curLabel].Insert(raNode2);
						exists = rgn_adj_list[bottomLabel].Insert(raNode1);

						//if the node already exists then place
						//nodes back onto the region adjacency
						//free list
						if (exists)
							free_rgn_adj_lists = oldRAFreeList;

					}
				}

				//check only to the right neighbors of the bottom boundary
				//pixels...

				//check the right for pixel locations whose x < width - 1
				for (j = 0; j < width - 1; j++)
				{
					//calculate pixel labels (i = height-1)
					curLabel = labels[i*width + j];	//current pixel
					rightLabel = labels[i*width + j + 1];	//right   pixel

															//check to the right, if the label of
															//the right pixel is not the same as that
															//of the current one then region[j] and region[j+1]
															//are adjacent to one another - update the RAM
					if (curLabel != rightLabel)
					{
						//obtain RAList object from region adjacency free
						//list
						raNode1 = free_rgn_adj_lists;
						raNode2 = free_rgn_adj_lists->next;

						//keep a pointer to the old region adj. free
						//list just in case nodes already exist in respective
						//region lists
						oldRAFreeList = free_rgn_adj_lists;

						//update region adjacency free list
						free_rgn_adj_lists = free_rgn_adj_lists->next->next;

						//populate RAList nodes
						raNode1->label = curLabel;
						raNode2->label = rightLabel;

						//insert nodes into the RAM
						exists = 0;
						rgn_adj_list[curLabel].Insert(raNode2);
						exists = rgn_adj_list[rightLabel].Insert(raNode1);

						//if the node already exists then place
						//nodes back onto the region adjacency
						//free list
						if (exists)
							free_rgn_adj_lists = oldRAFreeList;

					}

				}
			}

			~RegionAdjacencyTable()
			{

			}
		};

		void NewOptimizedFilter2(const cv::Mat& inp, MeanShiftSegmentationState& state)
		{
			int width = inp.cols;
			int height = inp.rows;
			int N = inp.channels();
			float sigma_s = static_cast<float>(sigma_s_);

			const float* data = inp.ptr<float>();
			float* msRawData = state.image.ptr<float>();

			// Declare Variables
			int		iterationCount, i, j, k, modeCandidateX, modeCandidateY, modeCandidate_i;
			double	mvAbs, diff, el;
			int L = width * height;

			//re-assign bandwidths to sigmaS and sigmaR
			if (((state.h[0] = sigma_s) <= 0) || ((state.h[1] = sigma_r_) <= 0))
				throw std::exception("sigmaS and/or sigmaR is zero or negative.");

			//define input data dimension with lattice
			int lN = N + 2;

			// Traverse each data point applying mean shift
			// to each data point

			// Allcocate memory for yk
			std::vector<double> yk(lN);

			// Allocate memory for Mh
			std::vector<double> Mh(lN);

			// let's use some temporary data
			std::vector<float> sdata(lN*L);

			// copy the scaled data
			int idxs, idxd;
			idxs = idxd = 0;
			if (N == 3)
			{
				for (i = 0; i<L; i++)
				{
					sdata[idxs++] = (i%width) / sigma_s;
					sdata[idxs++] = (i / width) / sigma_s;
					sdata[idxs++] = data[idxd++] / sigma_r_;
					sdata[idxs++] = data[idxd++] / sigma_r_;
					sdata[idxs++] = data[idxd++] / sigma_r_;
				}
			}
			else if (N == 1)
			{
				for (i = 0; i<L; i++)
				{
					sdata[idxs++] = (i%width) / sigma_s;
					sdata[idxs++] = (i / width) / sigma_s;
					sdata[idxs++] = data[idxd++] / sigma_r_;
				}
			}
			else
			{
				for (i = 0; i<L; i++)
				{
					sdata[idxs++] = (i%width) / sigma_s;
					sdata[idxs++] = (i / width) / sigma_s;
					for (j = 0; j<N; j++)
						sdata[idxs++] = data[idxd++] / sigma_r_;
				}
			}
			// index the data in the 3d buckets (x, y, L)
			std::vector<int> slist(L);
			int bucNeigh[27];

			float sMins; // just for L
			float sMaxs[3]; // for all
			sMaxs[0] = width / sigma_s;
			sMaxs[1] = height / sigma_s;
			sMins = sMaxs[2] = sdata[2];
			idxs = 2;
			float cval;
			for (i = 0; i<L; i++)
			{
				cval = sdata[idxs];
				if (cval < sMins)
					sMins = cval;
				else if (cval > sMaxs[2])
					sMaxs[2] = cval;

				idxs += lN;
			}

			int nBuck1, nBuck2, nBuck3;
			int cBuck1, cBuck2, cBuck3, cBuck;
			nBuck1 = (int)(sMaxs[0] + 3);
			nBuck2 = (int)(sMaxs[1] + 3);
			nBuck3 = (int)(sMaxs[2] - sMins + 3);
			std::vector<int> buckets(nBuck1*nBuck2*nBuck3);
			for (i = 0; i<(nBuck1*nBuck2*nBuck3); i++)
				buckets[i] = -1;

			idxs = 0;
			for (i = 0; i<L; i++)
			{
				// find bucket for current data and add it to the list
				cBuck1 = (int)sdata[idxs] + 1;
				cBuck2 = (int)sdata[idxs + 1] + 1;
				cBuck3 = (int)(sdata[idxs + 2] - sMins) + 1;
				cBuck = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);

				slist[i] = buckets[cBuck];
				buckets[cBuck] = i;

				idxs += lN;
			}
			// init bucNeigh
			idxd = 0;
			for (cBuck1 = -1; cBuck1 <= 1; cBuck1++)
			{
				for (cBuck2 = -1; cBuck2 <= 1; cBuck2++)
				{
					for (cBuck3 = -1; cBuck3 <= 1; cBuck3++)
					{
						bucNeigh[idxd++] = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);
					}
				}
			}
			double wsuml, weight;
			double hiLTr = 80.0 / sigma_r_;
			// done indexing/hashing


			// Initialize mode table used for basin of attraction
			memset(&(state.mode_table[0]), 0, width*height);

			for (i = 0; i < L; i++)
			{
				// if a mode was already assigned to this data point
				// then skip this point, otherwise proceed to
				// find its mode by applying mean shift...
				if (state.mode_table[i] == 1)
					continue;

				// initialize point list...
				state.point_count = 0;

				// Assign window center (window centers are
				// initialized by createLattice to be the point
				// data[i])
				idxs = i*lN;
				for (j = 0; j<lN; j++)
					yk[j] = sdata[idxs + j];

				// Calculate the mean shift vector using the lattice
				// LatticeMSVector(Mh, yk); // modify to new

				// Initialize mean shift vector
				for (j = 0; j < lN; j++)
					Mh[j] = 0;
				wsuml = 0;
				// uniformLSearch(Mh, yk_ptr); // modify to new
				// find bucket of yk
				cBuck1 = (int)yk[0] + 1;
				cBuck2 = (int)yk[1] + 1;
				cBuck3 = (int)(yk[2] - sMins) + 1;
				cBuck = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);
				for (j = 0; j<27; j++)
				{
					idxd = buckets[cBuck + bucNeigh[j]];
					// list parse, crt point is cHeadList
					while (idxd >= 0)
					{
						idxs = lN*idxd;
						// determine if inside search window
						el = sdata[idxs + 0] - yk[0];
						diff = el*el;
						el = sdata[idxs + 1] - yk[1];
						diff += el*el;

						if (diff < 1.0)
						{
							el = sdata[idxs + 2] - yk[2];
							if (yk[2] > hiLTr)
								diff = 4 * el*el;
							else
								diff = el*el;

							if (N>1)
							{
								el = sdata[idxs + 3] - yk[3];
								diff += el*el;
								el = sdata[idxs + 4] - yk[4];
								diff += el*el;
							}

							if (diff < 1.0)
							{
								weight = 1.0;
								for (k = 0; k<lN; k++)
									Mh[k] += weight*sdata[idxs + k];
								wsuml += weight;

								//set basin of attraction mode table
								if (diff < speed_threshold_)
								{
									if (state.mode_table[idxd] == 0)
									{
										state.point_list[state.point_count++] = idxd;
										state.mode_table[idxd] = 2;
									}
								}
							}
						}
						idxd = slist[idxd];
					}
				}
				if (wsuml > 0)
				{
					for (j = 0; j < lN; j++)
						Mh[j] = Mh[j] / wsuml - yk[j];
				}
				else
				{
					for (j = 0; j < lN; j++)
						Mh[j] = 0;
				}

				// Calculate its magnitude squared
				//mvAbs = 0;
				//for(j = 0; j < lN; j++)
				//	mvAbs += Mh[j]*Mh[j];
				mvAbs = (Mh[0] * Mh[0] + Mh[1] * Mh[1])*sigma_s_*sigma_s_;
				if (N == 3)
					mvAbs += (Mh[2] * Mh[2] + Mh[3] * Mh[3] + Mh[4] * Mh[4])*sigma_r_*sigma_r_;
				else
					mvAbs += Mh[2] * Mh[2] * sigma_r_*sigma_r_;


				// Keep shifting window center until the magnitude squared of the
				// mean shift vector calculated at the window center location is
				// under a specified threshold (Epsilon)

				// NOTE: iteration count is for speed up purposes only - it
				//       does not have any theoretical importance
				iterationCount = 1;
				while ((mvAbs >= EPSILON) && (iterationCount < LIMIT))
				{

					// Shift window location
					for (j = 0; j < lN; j++)
						yk[j] += Mh[j];

					// check to see if the current mode location is in the
					// basin of attraction...

					// calculate the location of yk on the lattice
					modeCandidateX = (int)(sigma_s_*yk[0] + 0.5);
					modeCandidateY = (int)(sigma_s_*yk[1] + 0.5);
					modeCandidate_i = modeCandidateY*width + modeCandidateX;

					// if mvAbs != 0 (yk did indeed move) then check
					// location basin_i in the mode table to see if
					// this data point either:

					// (1) has not been associated with a mode yet
					//     (modeTable[basin_i] = 0), so associate
					//     it with this one
					//
					// (2) it has been associated with a mode other
					//     than the one that this data point is converging
					//     to (modeTable[basin_i] = 1), so assign to
					//     this data point the same mode as that of basin_i

					if ((state.mode_table[modeCandidate_i] != 2) && (modeCandidate_i != i))
					{
						// obtain the data point at basin_i to
						// see if it is within h*TC_DIST_FACTOR of
						// of yk
						diff = 0;
						idxs = lN*modeCandidate_i;
						for (k = 2; k<lN; k++)
						{
							el = sdata[idxs + k] - yk[k];
							diff += el*el;
						}

						// if the data point at basin_i is within
						// a distance of h*TC_DIST_FACTOR of yk
						// then depending on modeTable[basin_i] perform
						// either (1) or (2)
						if (diff < speed_threshold_)
						{
							// if the data point at basin_i has not
							// been associated to a mode then associate
							// it with the mode that this one will converge
							// to
							if (state.mode_table[modeCandidate_i] == 0)
							{
								// no mode associated yet so associate
								// it with this one...
								state.point_list[state.point_count++] = modeCandidate_i;
								state.mode_table[modeCandidate_i] = 2;

							}
							else
							{

								// the mode has already been associated with
								// another mode, thererfore associate this one
								// mode and the modes in the point list with
								// the mode associated with data[basin_i]...

								// store the mode info into yk using msRawData...
								for (j = 0; j < N; j++)
									yk[j + 2] = msRawData[modeCandidate_i*N + j] / sigma_r_;

								// update mode table for this data point
								// indicating that a mode has been associated
								// with it
								state.mode_table[i] = 1;

								// indicate that a mode has been associated
								// to this data point (data[i])
								mvAbs = -1;

								// stop mean shift calculation...
								break;
							}
						}
					}

					// Calculate the mean shift vector at the new
					// window location using lattice
					// Calculate the mean shift vector using the lattice
					// LatticeMSVector(Mh, yk); // modify to new

					// Initialize mean shift vector
					for (j = 0; j < lN; j++)
						Mh[j] = 0;
					wsuml = 0;
					// uniformLSearch(Mh, yk_ptr); // modify to new
					// find bucket of yk
					cBuck1 = (int)yk[0] + 1;
					cBuck2 = (int)yk[1] + 1;
					cBuck3 = (int)(yk[2] - sMins) + 1;
					cBuck = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);
					for (j = 0; j<27; j++)
					{
						idxd = buckets[cBuck + bucNeigh[j]];
						// list parse, crt point is cHeadList
						while (idxd >= 0)
						{
							idxs = lN*idxd;
							// determine if inside search window
							el = sdata[idxs + 0] - yk[0];
							diff = el*el;
							el = sdata[idxs + 1] - yk[1];
							diff += el*el;

							if (diff < 1.0)
							{
								el = sdata[idxs + 2] - yk[2];
								if (yk[2] > hiLTr)
									diff = 4 * el*el;
								else
									diff = el*el;

								if (N>1)
								{
									el = sdata[idxs + 3] - yk[3];
									diff += el*el;
									el = sdata[idxs + 4] - yk[4];
									diff += el*el;
								}

								if (diff < 1.0)
								{
									weight = 1;
									for (k = 0; k<lN; k++)
										Mh[k] += weight*sdata[idxs + k];
									wsuml += weight;

									//set basin of attraction mode table
									if (diff < speed_threshold_)
									{
										if (state.mode_table[idxd] == 0)
										{
											state.point_list[state.point_count++] = idxd;
											state.mode_table[idxd] = 2;
										}
									}

								}
							}
							idxd = slist[idxd];
						}
					}
					if (wsuml > 0)
					{
						for (j = 0; j < lN; j++)
							Mh[j] = Mh[j] / wsuml - yk[j];
					}
					else
					{
						for (j = 0; j < lN; j++)
							Mh[j] = 0;
					}

					// Calculate its magnitude squared
					//mvAbs = 0;
					//for(j = 0; j < lN; j++)
					//	mvAbs += Mh[j]*Mh[j];
					mvAbs = (Mh[0] * Mh[0] + Mh[1] * Mh[1])*sigma_s_*sigma_s_;
					if (N == 3)
						mvAbs += (Mh[2] * Mh[2] + Mh[3] * Mh[3] + Mh[4] * Mh[4])*sigma_r_*sigma_r_;
					else
						mvAbs += Mh[2] * Mh[2] * sigma_r_*sigma_r_;

					// Increment iteration count
					iterationCount++;

				}

				// if a mode was not associated with this data point
				// yet associate it with yk...
				if (mvAbs >= 0)
				{
					// Shift window location
					for (j = 0; j < lN; j++)
						yk[j] += Mh[j];

					// update mode table for this data point
					// indicating that a mode has been associated
					// with it
					state.mode_table[i] = 1;

				}

				for (k = 0; k<N; k++)
					yk[k + 2] *= sigma_r_;

				// associate the data point indexed by
				// the point list with the mode stored
				// by yk
				for (j = 0; j < state.point_count; j++)
				{
					// obtain the point location from the
					// point list
					modeCandidate_i = state.point_list[j];

					// update the mode table for this point
					state.mode_table[modeCandidate_i] = 1;

					//store result into msRawData...
					for (k = 0; k < N; k++)
						msRawData[N*modeCandidate_i + k] = (float)(yk[k + 2]);
				}

				//store result into msRawData...
				for (j = 0; j < N; j++)
					msRawData[N*i + j] = (float)(yk[j + 2]);
			}
		}

		void NewOptimizedFilter1(const cv::Mat& inp, MeanShiftSegmentationState& state)
		{
			// Declare Variables
			int		iterationCount, i, j, k, modeCandidateX, modeCandidateY, modeCandidate_i;
			double	mvAbs, diff, el;

			int width = state.image.cols;
			int height = state.image.rows;
			int L = height * width;

			//re-assign bandwidths to sigmaS and sigmaR
			if (((state.h[0] = static_cast<float>(sigma_s_)) <= 0) || ((state.h[1] = sigma_r_) <= 0))
				throw std::exception("sigmaS and/or sigmaR is zero or negative.");

			//define input data dimension with lattice
			int lN = state.image.channels() + 2;

			// Traverse each data point applying mean shift
			// to each data point

			// Allcocate memory for yk
			std::vector<double> yk(lN);

			// Allocate memory for Mh
			std::vector<double> Mh(lN);

			const float* data = inp.ptr<float>();
			float* msRawData = state.image.ptr<float>();

			// let's use some temporary data
			std::vector<float> sdata(lN*L);

			// copy the scaled data
			int idxs, idxd;
			idxs = idxd = 0;
			if (inp.channels() == 3)
			{
				for (i = 0; i<L; i++)
				{
					sdata[idxs++] = (i%width) / static_cast<float>(sigma_s_);
					sdata[idxs++] = (i / width) / static_cast<float>(sigma_s_);
					sdata[idxs++] = data[idxd++] / sigma_r_;
					sdata[idxs++] = data[idxd++] / sigma_r_;
					sdata[idxs++] = data[idxd++] / sigma_r_;
				}
			}
			else if (inp.channels() == 1)
			{
				for (i = 0; i<L; i++)
				{
					sdata[idxs++] = (i%width) / static_cast<float>(sigma_s_);
					sdata[idxs++] = (i / width) / static_cast<float>(sigma_s_);
					sdata[idxs++] = data[idxd++] / sigma_r_;
				}
			}
			else
			{
				for (i = 0; i<L; i++)
				{
					sdata[idxs++] = (i%width) / static_cast<float>(sigma_s_);
					sdata[idxs++] = (i / width) / static_cast<float>(sigma_s_);
					for (j = 0; j<inp.channels(); j++)
						sdata[idxs++] = data[idxd++] / sigma_r_;
				}
			}
			// index the data in the 3d buckets (x, y, L)
			std::vector<int> slist(L);
			int bucNeigh[27];

			float sMins; // just for L
			float sMaxs[3]; // for all
			sMaxs[0] = width / static_cast<float>(sigma_s_);
			sMaxs[1] = height / static_cast<float>(sigma_s_);
			sMins = sMaxs[2] = sdata[2];
			idxs = 2;
			float cval;
			for (i = 0; i<L; i++)
			{
				cval = sdata[idxs];
				if (cval < sMins)
					sMins = cval;
				else if (cval > sMaxs[2])
					sMaxs[2] = cval;

				idxs += lN;
			}

			int nBuck1, nBuck2, nBuck3;
			int cBuck1, cBuck2, cBuck3, cBuck;
			nBuck1 = (int)(sMaxs[0] + 3);
			nBuck2 = (int)(sMaxs[1] + 3);
			nBuck3 = (int)(sMaxs[2] - sMins + 3);
			std::vector<int> buckets(nBuck1*nBuck2*nBuck3);
			for (i = 0; i<(nBuck1*nBuck2*nBuck3); i++)
				buckets[i] = -1;

			idxs = 0;
			for (i = 0; i<L; i++)
			{
				// find bucket for current data and add it to the list
				cBuck1 = (int)sdata[idxs] + 1;
				cBuck2 = (int)sdata[idxs + 1] + 1;
				cBuck3 = (int)(sdata[idxs + 2] - sMins) + 1;
				cBuck = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);

				slist[i] = buckets[cBuck];
				buckets[cBuck] = i;

				idxs += lN;
			}
			// init bucNeigh
			idxd = 0;
			for (cBuck1 = -1; cBuck1 <= 1; cBuck1++)
			{
				for (cBuck2 = -1; cBuck2 <= 1; cBuck2++)
				{
					for (cBuck3 = -1; cBuck3 <= 1; cBuck3++)
					{
						bucNeigh[idxd++] = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);
					}
				}
			}
			double wsuml;// , weight;
			double hiLTr = 80.0 / sigma_r_;
			// done indexing/hashing


			// Initialize mode table used for basin of attraction
			memset(&(state.mode_table[0]), 0, width*height);

			for (i = 0; i < L; i++)
			{
				// if a mode was already assigned to this data point
				// then skip this point, otherwise proceed to
				// find its mode by applying mean shift...
				if (state.mode_table[i] == 1)
					continue;

				// initialize point list...
				state.point_count = 0;

				// Assign window center (window centers are
				// initialized by createLattice to be the point
				// data[i])
				idxs = i*lN;
				for (j = 0; j<lN; j++)
					yk[j] = sdata[idxs + j];

				// Calculate the mean shift vector using the lattice
				// LatticeMSVector(Mh, yk); // modify to new
				// Initialize mean shift vector
				for (j = 0; j < lN; j++)
					Mh[j] = 0;
				wsuml = 0;
				// uniformLSearch(Mh, yk_ptr); // modify to new
				// find bucket of yk
				cBuck1 = (int)yk[0] + 1;
				cBuck2 = (int)yk[1] + 1;
				cBuck3 = (int)(yk[2] - sMins) + 1;
				cBuck = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);
				for (j = 0; j<27; j++)
				{
					idxd = buckets[cBuck + bucNeigh[j]];
					// list parse, crt point is cHeadList
					while (idxd >= 0)
					{
						idxs = lN*idxd;
						// determine if inside search window
						el = sdata[idxs + 0] - yk[0];
						diff = el*el;
						el = sdata[idxs + 1] - yk[1];
						diff += el*el;

						if (diff < 1.0)
						{
							el = sdata[idxs + 2] - yk[2];
							if (yk[2] > hiLTr)
								diff = 4 * el*el;
							else
								diff = el*el;

							if (inp.channels()>1)
							{
								el = sdata[idxs + 3] - yk[3];
								diff += el*el;
								el = sdata[idxs + 4] - yk[4];
								diff += el*el;
							}

							//weightMap
							if (diff < 1.0)
							{
								double weight = 1.0;
								for (k = 0; k<lN; k++)
									Mh[k] += weight*sdata[idxs + k];
								wsuml += weight;
							}
						}
						idxd = slist[idxd];
					}
				}
				if (wsuml > 0)
				{
					for (j = 0; j < lN; j++)
						Mh[j] = Mh[j] / wsuml - yk[j];
				}
				else
				{
					for (j = 0; j < lN; j++)
						Mh[j] = 0;
				}
				// Calculate its magnitude squared
				//mvAbs = 0;
				//for(j = 0; j < lN; j++)
				//	mvAbs += Mh[j]*Mh[j];
				mvAbs = (Mh[0] * Mh[0] + Mh[1] * Mh[1])*sigma_s_*sigma_s_;
				if (inp.channels() == 3)
					mvAbs += (Mh[2] * Mh[2] + Mh[3] * Mh[3] + Mh[4] * Mh[4])*sigma_r_*sigma_r_;
				else
					mvAbs += Mh[2] * Mh[2] * sigma_r_*sigma_r_;


				// Keep shifting window center until the magnitude squared of the
				// mean shift vector calculated at the window center location is
				// under a specified threshold (Epsilon)

				// NOTE: iteration count is for speed up purposes only - it
				//       does not have any theoretical importance
				iterationCount = 1;
				while ((mvAbs >= EPSILON) && (iterationCount < LIMIT))
				{

					// Shift window location
					for (j = 0; j < lN; j++)
						yk[j] += Mh[j];

					// check to see if the current mode location is in the
					// basin of attraction...

					// calculate the location of yk on the lattice
					modeCandidateX = (int)(sigma_s_*yk[0] + 0.5);
					modeCandidateY = (int)(sigma_s_*yk[1] + 0.5);
					modeCandidate_i = modeCandidateY*width + modeCandidateX;

					// if mvAbs != 0 (yk did indeed move) then check
					// location basin_i in the mode table to see if
					// this data point either:

					// (1) has not been associated with a mode yet
					//     (modeTable[basin_i] = 0), so associate
					//     it with this one
					//
					// (2) it has been associated with a mode other
					//     than the one that this data point is converging
					//     to (modeTable[basin_i] = 1), so assign to
					//     this data point the same mode as that of basin_i

					if ((state.mode_table[modeCandidate_i] != 2) && (modeCandidate_i != i))
					{
						// obtain the data point at basin_i to
						// see if it is within h*TC_DIST_FACTOR of
						// of yk
						diff = 0;
						idxs = lN*modeCandidate_i;
						for (k = 2; k<lN; k++)
						{
							el = sdata[idxs + k] - yk[k];
							diff += el*el;
						}

						// if the data point at basin_i is within
						// a distance of h*TC_DIST_FACTOR of yk
						// then depending on modeTable[basin_i] perform
						// either (1) or (2)
						if (diff < TC_DIST_FACTOR)
						{
							// if the data point at basin_i has not
							// been associated to a mode then associate
							// it with the mode that this one will converge
							// to
							if (state.mode_table[modeCandidate_i] == 0)
							{
								// no mode associated yet so associate
								// it with this one...
								state.point_list[state.point_count++] = modeCandidate_i;
								state.mode_table[modeCandidate_i] = 2;

							}
							else
							{

								// the mode has already been associated with
								// another mode, thererfore associate this one
								// mode and the modes in the point list with
								// the mode associated with data[basin_i]...

								// store the mode info into yk using msRawData...
								for (j = 0; j < inp.channels(); j++)
									yk[j + 2] = msRawData[modeCandidate_i*inp.channels() + j] / sigma_r_;

								// update mode table for this data point
								// indicating that a mode has been associated
								// with it
								state.mode_table[i] = 1;

								// indicate that a mode has been associated
								// to this data point (data[i])
								mvAbs = -1;

								// stop mean shift calculation...
								break;
							}
						}
					}

					// Calculate the mean shift vector at the new
					// window location using lattice
					// Calculate the mean shift vector using the lattice
					// LatticeMSVector(Mh, yk); // modify to new
					// Initialize mean shift vector
					for (j = 0; j < lN; j++)
						Mh[j] = 0;
					wsuml = 0;
					// uniformLSearch(Mh, yk_ptr); // modify to new
					// find bucket of yk
					cBuck1 = (int)yk[0] + 1;
					cBuck2 = (int)yk[1] + 1;
					cBuck3 = (int)(yk[2] - sMins) + 1;
					cBuck = cBuck1 + nBuck1*(cBuck2 + nBuck2*cBuck3);
					for (j = 0; j<27; j++)
					{
						idxd = buckets[cBuck + bucNeigh[j]];
						// list parse, crt point is cHeadList
						while (idxd >= 0)
						{
							idxs = lN*idxd;
							// determine if inside search window
							el = sdata[idxs + 0] - yk[0];
							diff = el*el;
							el = sdata[idxs + 1] - yk[1];
							diff += el*el;

							if (diff < 1.0)
							{
								el = sdata[idxs + 2] - yk[2];
								if (yk[2] > hiLTr)
									diff = 4 * el*el;
								else
									diff = el*el;

								if (inp.channels()>1)
								{
									el = sdata[idxs + 3] - yk[3];
									diff += el*el;
									el = sdata[idxs + 4] - yk[4];
									diff += el*el;
								}
								//weightMap
								if (diff < 1.0)
								{
									double weight = 1.0;
									for (k = 0; k<lN; k++)
										Mh[k] += weight*sdata[idxs + k];
									wsuml += weight;
								}
							}
							idxd = slist[idxd];
						}
					}
					if (wsuml > 0)
					{
						for (j = 0; j < lN; j++)
							Mh[j] = Mh[j] / wsuml - yk[j];
					}
					else
					{
						for (j = 0; j < lN; j++)
							Mh[j] = 0;
					}

					// Calculate its magnitude squared
					//mvAbs = 0;
					//for(j = 0; j < lN; j++)
					//	mvAbs += Mh[j]*Mh[j];
					mvAbs = (Mh[0] * Mh[0] + Mh[1] * Mh[1])*sigma_s_*sigma_s_;
					if (inp.channels() == 3)
						mvAbs += (Mh[2] * Mh[2] + Mh[3] * Mh[3] + Mh[4] * Mh[4])*sigma_r_*sigma_r_;
					else
						mvAbs += Mh[2] * Mh[2] * sigma_r_*sigma_r_;

					// Increment iteration count
					iterationCount++;

				}

				// if a mode was not associated with this data point
				// yet associate it with yk...
				if (mvAbs >= 0)
				{
					// Shift window location
					for (j = 0; j < lN; j++)
						yk[j] += Mh[j];

					// update mode table for this data point
					// indicating that a mode has been associated
					// with it
					state.mode_table[i] = 1;

				}

				for (k = 0; k<inp.channels(); k++)
					yk[k + 2] *= sigma_r_;

				// associate the data point indexed by
				// the point list with the mode stored
				// by yk
				for (j = 0; j < state.point_count; j++)
				{
					// obtain the point location from the
					// point list
					modeCandidate_i = state.point_list[j];

					// update the mode table for this point
					state.mode_table[modeCandidate_i] = 1;

					//store result into msRawData...
					for (k = 0; k < inp.channels(); k++)
						msRawData[inp.channels()*modeCandidate_i + k] = (float)(yk[k + 2]);
				}

				//store result into msRawData...
				for (j = 0; j < inp.channels(); j++)
					msRawData[inp.channels()*i + j] = (float)(yk[j + 2]);

			}
		}

		void Fill(int regionLoc, int label, const std::vector<int>& neigh, MeanShiftSegmentationState& state)
		{
			float* LUV_data = state.image.ptr<float>();
			int* labels = state.labels.ptr<int>();
			int width = state.image.cols;
			int height = state.image.rows;
			int N = state.image.channels();

			//declare variables
			int	i, k, neighLoc, neighborsFound, imageSize = width*height;

			//Fill region starting at region location
			//using labels...

			//initialzie indexTable
			int	index = 0;
			state.index_table[0] = regionLoc;

			//increment mode point counts for this region to
			//indicate that one pixel belongs to this region
			state.mode_point_counts[label]++;

			while (true)
			{

				//assume no neighbors will be found
				neighborsFound = 0;

				//check the eight connected neighbors at regionLoc -
				//if a pixel has similar color to that located at 
				//regionLoc then declare it as part of this region
				for (i = 0; i < static_cast<int>(neigh.size()); i++)  
				{
					//check bounds and if neighbor has been already labeled
					neighLoc = regionLoc + neigh[i];
					if ((neighLoc >= 0) && (neighLoc < imageSize) && (labels[neighLoc] < 0))
					{
						for (k = 0; k < N; k++)
						{
							//					if(LUV_data[(regionLoc*N)+k] != LUV_data[(neighLoc*N)+k])
							if (fabs(LUV_data[(regionLoc*N) + k] - LUV_data[(neighLoc*N) + k]) >= LUV_THRESHOLD)
								break;
						}

						//neighbor i belongs to this region so label it and
						//place it onto the index table buffer for further
						//processing
						if (k == N)
						{
							//assign label to neighbor i
							labels[neighLoc] = label;

							//increment region point count
							state.mode_point_counts[label]++;

							//place index of neighbor i onto the index tabel buffer
							state.index_table[++index] = neighLoc;

							//indicate that a neighboring region pixel was
							//identified
							neighborsFound = 1;
						}
					}
				}

				//check the indexTable to see if there are any more
				//entries to be explored - if so explore them, otherwise
				//exit the loop - we are finished
				if (neighborsFound)
					regionLoc = state.index_table[index];
				else if (index > 1)
					regionLoc = state.index_table[--index];
				else
					break; //fill complete
			}

		}

		void Connect(MeanShiftSegmentationState& state, int connectivity)
		{
			int N = state.image.channels();
			int width = state.image.cols;
			int height = state.image.rows;
			int* labels = state.labels.ptr<int>();
			float* LUV_data = state.image.ptr<float>();

			//define eight connected neighbors
			std::vector<int> neigh;
			if (connectivity == 8) {
				neigh.resize(8);
				neigh[0] = 1;
				neigh[1] = 1 - width;
				neigh[2] = -width;
				neigh[3] = -(1 + width);
				neigh[4] = -1;
				neigh[5] = width - 1;
				neigh[6] = width;
				neigh[7] = width + 1;
			} else {
				neigh.resize(4);
				neigh[0] = 1;
				neigh[1] = -width;
				neigh[2] = -1;
				neigh[3] = width;
			}

			//initialize labels and modePointCounts
			int i;
			for (i = 0; i < width*height; i++)
			{
				labels[i] = -1;
				state.mode_point_counts[i] = 0;
			}

			//Traverse the image labeling each new region encountered
			int k, label = -1;
			for (i = 0; i < height*width; i++)
			{
				//if this region has not yet been labeled - label it
				if (labels[i] < 0)
				{
					//assign new label to this region
					labels[i] = ++label;

					//copy region color into modes
					for (k = 0; k < N; k++)
						state.modes[(N*label) + k] = LUV_data[(N*i) + k];
					//				modes[(N*label)+k]	= (float)(LUV_data[(N*i)+k]);

					//populate labels with label for this specified region
					//calculating modePointCounts[label]...
					// TODO: this could be replaced with opencv's floodfill and then state.index_table could
					// be removed from MeanShiftSegmentationState.
					Fill(i, label, neigh, state);
				}
			}

			//calculate region count using label
			state.rgn_count = label + 1;
		}

		bool InWindow(const MeanShiftSegmentationState& state, int mode1, int mode2)
		{
			int		k = 1, s = 0, p;
			int N = state.image.channels();
			int P[2] = { 2, N };
			double	diff = 0, el;
			while ((diff < 0.25) && (k != KP)) // Partial Distortion Search
			{
				//Calculate distance squared of sub-space s	
				diff = 0;
				for (p = 0; p < P[k]; p++)
				{
					el = (state.modes[mode1*N + p + s] - state.modes[mode2*N + p + s]) / (state.h[k]);
					if ((!p) && (k == 1) && (state.modes[mode1*N] > 80))
						diff += 4 * el*el;
					else
						diff += el*el;
				}

				//next subspace
				s += P[k];
				k++;
			}
			return (bool)(diff < 0.25);
		}

		void TransitiveClosure(MeanShiftSegmentationState& state)
		{
			int width = state.image.cols;
			int height = state.image.rows;
			int N = state.image.channels();
			const float epsilon = 1.0f;
			//Step (1):

			// Build RAM using classifiction structure originally
			// generated by the method GridTable::Connect()
			RegionAdjacencyTable ram(state);

			//Step (2):

			//Treat each region Ri as a disjoint set:

			// - attempt to join Ri and Rj for all i != j that are neighbors and
			//   whose associated modes are a normalized distance of < 0.5 from one
			//   another

			// - the label of each region in the raList is treated as a pointer to the
			//   canonical element of that region (e.g. raList[i], initially has raList[i].label = i,
			//   namely each region is initialized to have itself as its canonical element).

			//Traverse RAM attempting to join raList[i] with its neighbors...
			int		i, iCanEl, neighCanEl;
			float	threshold;
			RgnAdjList	*neighbor;
			for (i = 0; i < state.rgn_count; i++)
			{
				//aquire first neighbor in region adjacency list pointed to
				//by raList[i]
				neighbor = ram.rgn_adj_list[i].next;

				//compute edge strenght threshold using global and local
				//epsilon
				if (epsilon > ram.rgn_adj_list[i].edge_strength)
					threshold = epsilon;
				else
					threshold = ram.rgn_adj_list[i].edge_strength;

				//traverse region adjacency list of region i, attempting to join
				//it with regions whose mode is a normalized distance < 0.5 from
				//that of region i...
				while (neighbor)
				{
					//attempt to join region and neighbor...
					if ((InWindow(state, i, neighbor->label)) && (neighbor->edge_strength < epsilon))
					{
						//region i and neighbor belong together so join them
						//by:

						// (1) find the canonical element of region i
						iCanEl = i;
						while (ram.rgn_adj_list[iCanEl].label != iCanEl)
							iCanEl = ram.rgn_adj_list[iCanEl].label;

						// (2) find the canonical element of neighboring region
						neighCanEl = neighbor->label;
						while (ram.rgn_adj_list[neighCanEl].label != neighCanEl)
							neighCanEl = ram.rgn_adj_list[neighCanEl].label;

						// if the canonical elements of are not the same then assign
						// the canonical element having the smaller label to be the parent
						// of the other region...
						if (iCanEl < neighCanEl)
							ram.rgn_adj_list[neighCanEl].label = iCanEl;
						else
						{
							//must replace the canonical element of previous
							//parent as well
							ram.rgn_adj_list[ram.rgn_adj_list[iCanEl].label].label = neighCanEl;

							//re-assign canonical element
							ram.rgn_adj_list[iCanEl].label = neighCanEl;
						}
					}

					//check the next neighbor...
					neighbor = neighbor->next;

				}
			}

			// Step (3):

			// Level binary trees formed by canonical elements
			for (i = 0; i < state.rgn_count; i++)
			{
				iCanEl = i;
				while (ram.rgn_adj_list[iCanEl].label != iCanEl)
					iCanEl = ram.rgn_adj_list[iCanEl].label;
				ram.rgn_adj_list[i].label = iCanEl;
			}

			// Step (4):

			//Traverse joint sets, relabeling image.

			// (a)

			// Accumulate modes and re-compute point counts using canonical
			// elements generated by step 2.

			//allocate memory for mode and point count temporary buffers...
			std::vector<float> modes_buffer(N*state.rgn_count);
			std::vector<int> MPC_buffer(state.rgn_count);

			//initialize buffers to zero
			for (i = 0; i < state.rgn_count; i++)
				MPC_buffer[i] = 0;
			for (i = 0; i < N * state.rgn_count; i++)
				modes_buffer[i] = 0;

			//traverse raList accumulating modes and point counts
			//using canoncial element information...
			int k, iMPC;
			for (i = 0; i < state.rgn_count; i++)
			{

				//obtain canonical element of region i
				iCanEl = ram.rgn_adj_list[i].label;

				//obtain mode point count of region i
				iMPC = state.mode_point_counts[i];

				//accumulate modes_buffer[iCanEl]
				for (k = 0; k < N; k++)
					modes_buffer[(N*iCanEl) + k] += iMPC * state.modes[(N*i) + k];

				//accumulate MPC_buffer[iCanEl]
				MPC_buffer[iCanEl] += iMPC;

			}

			// (b)

			// Re-label new regions of the image using the canonical
			// element information generated by step (2)

			// Also use this information to compute the modes of the newly
			// defined regions, and to assign new region point counts in
			// a consecute manner to the modePointCounts array

			//allocate memory for label buffer
			std::vector<int> label_buffer(state.rgn_count);

			//initialize label buffer to -1
			for (i = 0; i < state.rgn_count; i++)
				label_buffer[i] = -1;

			//traverse raList re-labeling the regions
			int	label = -1;
			for (i = 0; i < state.rgn_count; i++)
			{
				//obtain canonical element of region i
				iCanEl = ram.rgn_adj_list[i].label;
				if (label_buffer[iCanEl] < 0)
				{
					//assign a label to the new region indicated by canonical
					//element of i
					label_buffer[iCanEl] = ++label;

					//recompute mode storing the result in modes[label]...
					iMPC = MPC_buffer[iCanEl];
					for (k = 0; k < N; k++)
						state.modes[(N*label) + k] = (modes_buffer[(N*iCanEl) + k]) / (iMPC);

					//assign a corresponding mode point count for this region into
					//the mode point counts array using the MPC buffer...
					state.mode_point_counts[label] = MPC_buffer[iCanEl];
				}
			}

			//re-assign region count using label counter
			int	oldRegionCount = state.rgn_count;
			state.rgn_count = label + 1;

			// (c)

			// Use the label buffer to reconstruct the label map, which specified
			// the new image given its new regions calculated above
			int* labels = state.labels.ptr<int>();
			for (i = 0; i < height*width; i++)
				labels[i] = label_buffer[ram.rgn_adj_list[labels[i]].label];
		}

		float SqDistance(const MeanShiftSegmentationState& state, int mode1, int mode2)
		{
			int N = state.image.channels();
			int		k = 1, s = 0, p;
			float	dist = 0, el;
			int P[2] = { 2, N };
			for (k = 1; k < KP; k++)
			{
				//Calculate distance squared of sub-space s	
				for (p = 0; p < P[k]; p++)
				{
					el = (state.modes[mode1*N + p + s] - state.modes[mode2*N + p + s]) / (state.h[k]);
					dist += el*el;
				}

				//next subspace
				s += P[k];
				k++;
			}

			//return normalized square distance between modes
			//1 and 2
			return dist;
		}

		void Prune(MeanShiftSegmentationState& state)
		{
			int N = state.image.channels();
			//Allocate Memory for temporary buffers...

			//allocate memory for mode and point count temporary buffers...
			std::vector<float> modes_buffer(N*state.rgn_count);
			std::vector<int> MPC_buffer(state.rgn_count);

			//allocate memory for label buffer
			std::vector<int> label_buffer(state.rgn_count);

			//Declare variables
			int		i, k, candidate, iCanEl, neighCanEl, iMPC, label, oldRegionCount, minRegionCount;
			double	minSqDistance, neighborDistance;
			RgnAdjList	*neighbor;

			//Apply pruning algorithm to classification structure, removing all regions whose area
			//is under the threshold area minRegion (pixels)
			do
			{
				//Assume that no region has area under threshold area  of 
				minRegionCount = 0;

				//Step (1):
				// Build RAM using classifiction structure originally
				// generated by the method GridTable::Connect()
				RegionAdjacencyTable ram(state);

				// Step (2):
				// Traverse the RAM joining regions whose area is less than minRegion (pixels)
				// with its respective candidate region.
				// A candidate region is a region that displays the following properties:
				//	- it is adjacent to the region being pruned
				//  - the distance of its mode is a minimum to that of the region being pruned
				//    such that or it is the only adjacent region having an area greater than
				//    minRegion

				for (i = 0; i < state.rgn_count; i++)
				{
					//if the area of the ith region is less than minRegion
					//join it with its candidate region...

					//*******************************************************************************

					//Note: Adjust this if statement if a more sophisticated pruning criterion
					//      is desired. Basically in this step a region whose area is less than
					//      minRegion is pruned by joining it with its "closest" neighbor (in color).
					//      Therefore, by placing a different criterion for fusing a region the
					//      pruning method may be altered to implement a more sophisticated algorithm.

					//*******************************************************************************

					if (state.mode_point_counts[i] < min_size_)
					{
						//update minRegionCount to indicate that a region
						//having area less than minRegion was found
						minRegionCount++;

						//obtain a pointer to the first region in the
						//region adjacency list of the ith region...
						neighbor = ram.rgn_adj_list[i].next;

						//std::string dbg = "new => " + std::to_string(neighbor->label) + "\n";
						//OutputDebugStringA(dbg.c_str());

						//calculate the distance between the mode of the ith
						//region and that of the neighboring region...
						candidate = neighbor->label;
						minSqDistance = SqDistance(state, i, candidate);

						//traverse region adjacency list of region i and select
						//a candidate region
						neighbor = neighbor->next;
						while (neighbor)
						{

							//calculate the square distance between region i
							//and current neighbor...
							neighborDistance = SqDistance(state, i, neighbor->label);

							//if this neighbors square distance to region i is less
							//than minSqDistance, then select this neighbor as the
							//candidate region for region i
							if (neighborDistance < minSqDistance)
							{
								minSqDistance = neighborDistance;
								candidate = neighbor->label;
							}

							//traverse region list of region i
							neighbor = neighbor->next;

						}

						//join region i with its candidate region:

						// (1) find the canonical element of region i
						iCanEl = i;
						while (ram.rgn_adj_list[iCanEl].label != iCanEl)
							iCanEl = ram.rgn_adj_list[iCanEl].label;

						// (2) find the canonical element of neighboring region
						neighCanEl = candidate;
						while (ram.rgn_adj_list[neighCanEl].label != neighCanEl)
							neighCanEl = ram.rgn_adj_list[neighCanEl].label;

						// if the canonical elements of are not the same then assign
						// the canonical element having the smaller label to be the parent
						// of the other region...
						if (iCanEl < neighCanEl)
							ram.rgn_adj_list[neighCanEl].label = iCanEl;
						else
						{
							//must replace the canonical element of previous
							//parent as well
							ram.rgn_adj_list[ram.rgn_adj_list[iCanEl].label].label = neighCanEl;

							//re-assign canonical element
							ram.rgn_adj_list[iCanEl].label = neighCanEl;
						}
					}
				}

				// Step (3):

				// Level binary trees formed by canonical elements
				for (i = 0; i < state.rgn_count; i++)
				{
					iCanEl = i;
					while (ram.rgn_adj_list[iCanEl].label != iCanEl)
						iCanEl = ram.rgn_adj_list[iCanEl].label;
					ram.rgn_adj_list[i].label = iCanEl;
				}

				// Step (4):

				//Traverse joint sets, relabeling image.

				// Accumulate modes and re-compute point counts using canonical
				// elements generated by step 2.

				//initialize buffers to zero
				for (i = 0; i < state.rgn_count; i++)
					MPC_buffer[i] = 0;
				for (i = 0; i < N*state.rgn_count; i++)
					modes_buffer[i] = 0;

				//traverse raList accumulating modes and point counts
				//using canoncial element information...
				for (i = 0; i < state.rgn_count; i++)
				{
					//obtain canonical element of region i
					iCanEl = ram.rgn_adj_list[i].label;

					//obtain mode point count of region i
					iMPC = state.mode_point_counts[i];

					//accumulate modes_buffer[iCanEl]
					for (k = 0; k < N; k++)
						modes_buffer[(N*iCanEl) + k] += iMPC*state.modes[(N*i) + k];

					//accumulate MPC_buffer[iCanEl]
					MPC_buffer[iCanEl] += iMPC;

				}

				// (b)

				// Re-label new regions of the image using the canonical
				// element information generated by step (2)

				// Also use this information to compute the modes of the newly
				// defined regions, and to assign new region point counts in
				// a consecute manner to the modePointCounts array

				//initialize label buffer to -1
				for (i = 0; i < state.rgn_count; i++)
					label_buffer[i] = -1;

				//traverse raList re-labeling the regions
				label = -1;
				for (i = 0; i < state.rgn_count; i++)
				{
					//obtain canonical element of region i
					iCanEl = ram.rgn_adj_list[i].label;
					if (label_buffer[iCanEl] < 0)
					{
						//assign a label to the new region indicated by canonical
						//element of i
						label_buffer[iCanEl] = ++label;

						//recompute mode storing the result in modes[label]...
						iMPC = MPC_buffer[iCanEl];
						for (k = 0; k < N; k++)
							state.modes[(N*label) + k] = (modes_buffer[(N*iCanEl) + k]) / (iMPC);

						//assign a corresponding mode point count for this region into
						//the mode point counts array using the MPC buffer...
						state.mode_point_counts[label] = MPC_buffer[iCanEl];
					}
				}

				//re-assign region count using label counter
				oldRegionCount = state.rgn_count;
				state.rgn_count = label + 1;

				// (c)

				// Use the label buffer to reconstruct the label map, which specified
				// the new image given its new regions calculated above
				int width = state.image.cols;
				int height = state.image.rows;
				int* labels = state.labels.ptr<int>();
				for (i = 0; i < height*width; i++)
					labels[i] = label_buffer[ram.rgn_adj_list[labels[i]].label];

			} while (minRegionCount > 0);
		}

		void FuseRegions(MeanShiftSegmentationState& state)
		{
			int width = state.image.cols;
			int height = state.image.rows;
			int L = width * height;
			int N = state.image.channels();
			float sigmaS = sigma_r_;

			if ((state.h[1] = sigmaS) <= 0)
				throw std::exception("FuseRegions : The feature radius must be greater than or equal to zero.");

			//allocate memory visit table
			std::vector<uchar> visitTable(L);

			//Apply transitive closure iteratively to the regions classified
			//by the RAM updating labels and modes until the color of each neighboring
			//region is within sqrt(rR2) of one another.
			TransitiveClosure(state);
			int oldRC = state.rgn_count;
			int deltaRC, counter = 0;
			do {
				TransitiveClosure(state);
				deltaRC = oldRC - state.rgn_count;
				oldRC = state.rgn_count;
				counter++;
			} while ((deltaRC <= 0) && (counter < 10));

			//Prune spurious regions (regions whose area is under
			//minRegion) using RAM
			Prune(state);

			//output to msRawData
			const int* labels = state.labels.ptr<int>();
			float* msRawData = state.image.ptr<float>();
			int i, j, label;
			for (i = 0; i < L; i++)
			{
				label = labels[i];
				for (j = 0; j < N; j++)
				{
					msRawData[N*i + j] = state.modes[N*label + j];
				}
			}
		}

	public:
		MeanShiftSegmentationImpl() {
			sigma_r_ = 6.5;
			sigma_s_ = 7;
			optimized_ = true;
			min_size_ = 20;
			speed_threshold_ = 0.5f;
		}

		// Inherited via MeanShiftSegmentation
		virtual void processImage(cv::InputArray inp, cv::OutputArray segmented) override
		{
			cv::Mat labels;
			processImage(inp, segmented, labels);
		}

		virtual void processImage(cv::InputArray inp, cv::OutputArray segmented, cv::OutputArray labelMap) override
		{
			cv::Mat src = inp.getMat();
			if (src.channels() != 1 && src.channels() != 3)
				throw std::exception("MeanShiftSegmentation requires input with 1 or 3 channels.");

			cv::Mat luv = RgbToLuv(src);

			MeanShiftSegmentationState state(luv, segmented, labelMap);
			if (optimized_)
				NewOptimizedFilter2(luv, state);
			else
				NewOptimizedFilter1(luv, state);

			Connect(state, connectivity_);
			FuseRegions(state);

			segmented.create(src.rows, src.cols, inp.type());
			auto mat = segmented.getMat();
			LuvToRgb(state.image).copyTo(mat);
		}

		virtual void setConnectivity(int n) override {
			if (n != 8 && n != 4) {
				throw std::runtime_error("illegal connectivity param. must be 4 or 8.");
			}
			connectivity_ = n;
		}

		virtual int getConnectivity() const override {
			return connectivity_;
		}

		virtual void setSigmaS(int val) override
		{
			sigma_s_ = val;
		}

		virtual int getSigmaS() const override
		{
			return sigma_s_;
		}

		virtual void setSigmaR(float val) override
		{
			sigma_r_ = val;
		}

		virtual float getSigmaR() const override
		{
			return sigma_r_;
		}

		virtual void setMinSize(int min_size) override
		{
			min_size_ = min_size;
		}

		virtual int getMinSize() override
		{
			return min_size_;
		}

		virtual void setOptimized(bool val) override
		{
			optimized_ = val;
		}

		virtual bool getOptimized() const override
		{
			return optimized_;
		}
	};
}

cv::Ptr<MeanShiftSegmentation> createMeanShiftSegmentation(int sigmaS, float sigmaR, int min_size, int connectivity, bool optimized)
{
	cv::Ptr<MeanShiftSegmentation> mean_shift_seg = cv::makePtr<MeanShiftSegmentationImpl>();

	mean_shift_seg->setSigmaS(sigmaS);
	mean_shift_seg->setSigmaR(sigmaR);
	mean_shift_seg->setMinSize(min_size);
	mean_shift_seg->setOptimized(optimized);
	mean_shift_seg->setConnectivity(connectivity);

	return mean_shift_seg;
}
