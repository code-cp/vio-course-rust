use std::collections::{HashMap, BTreeMap}; 
use std::sync::{Arc, Mutex}; 
use std::fs::File;
use std::io::Write;
use nalgebra as na; 

use crate::vertex::*; 
use crate::edge::*; 
use crate::nalgebra_types::*; 

/// SLAM problem: pose and landmark are separated, Hessian is sparse 
/// Generic problem: Hessian is dense 
pub enum ProblemType {
    SlamProblem,
    GenericProblem,    
}

pub type HashVertex = BTreeMap<usize, Arc<Mutex<dyn Vertex>>>; 
pub type HashEdge = HashMap<usize, Arc<Mutex<dyn Edge>>>;
pub type HashVertexIdToEdge = HashMap<usize, Vec<Arc<Mutex<dyn Edge>>>>;

pub struct Problem {
    stop_threshold_lm: f64, 
    current_chi: f64, 
    current_lambda: f64, 
    ni: f64, 

    problem_type: ProblemType, 
    
    // information matrix 
    hessian: MatXX, 
    b: VecX, 
    delta_x: VecX,

    // prior 
    h_prior: MatXX, 
    b_prior: VecX, 
    jt_prior_inv: MatXX, 
    err_prior: VecX, 

    // pose in SBA 
    h_pp_schur: MatXX, 
    b_pp_schur: VecX,  

    // hessian 
    h_pp: MatXX, 
    b_pp: VecX, 
    h_ll: MatXX, 
    b_ll: VecX, 

    vertices: HashVertex, 
    edges: HashEdge, 
    vertex_to_edge: HashVertexIdToEdge, 

    // ordering 
    ordering_poses: usize, 
    ordering_landmarks: usize, 
    ordering_generic: usize,
    /// pose based on ordering   
    idx_pose_vertices: HashVertex, 
    /// landmark based on ordering 
    idx_landmark_vertices: HashVertex,
    /// vertices need to marg  
    vertices_marg: HashVertex,

    b_debug: bool, 
    t_hessian_cost: f32, 

    out_file: File,
    n_iter: usize,  
}

impl Problem {
    pub fn new(
        problem_type: ProblemType, 
    ) -> Self {
        let stop_threshold_lm = 0.0;
        let current_chi = 0.0;
        let current_lambda = 0.0; 
        let ni = 0.0;  

        let hessian = MatXX::zeros(0, 0);
        let b = VecX::zeros(0);
        let delta_x = VecX::zeros(0);

        let h_prior = MatXX::zeros(0, 0);
        let b_prior = VecX::zeros(0);
        let jt_prior_inv = MatXX::zeros(0, 0);
        let err_prior = VecX::zeros(0);

        let h_pp_schur = MatXX::zeros(0, 0);
        let b_pp_schur = VecX::zeros(0);

        let h_pp = MatXX::zeros(0, 0);
        let b_pp = VecX::zeros(0);
        let h_ll = MatXX::zeros(0, 0);
        let b_ll = VecX::zeros(0);

        let vertices = BTreeMap::new();
        let edges = HashMap::new();
        let vertex_to_edge = HashMap::new();

        let ordering_poses = 0;
        let ordering_landmarks = 0;
        let ordering_generic = 0;

        let idx_pose_vertices = BTreeMap::new();
        let idx_landmark_vertices = BTreeMap::new();
        let vertices_marg = BTreeMap::new();

        let b_debug = false;
        let t_hessian_cost = 0.0;

        let out_file = File::create("data_mu.txt").expect("Should be able to create output file");

        let n_iter = 0;

        Self {
            stop_threshold_lm, 
            current_chi, 
            current_lambda, 
            ni, 
            problem_type,
            hessian,
            b,
            delta_x,
            h_prior,
            b_prior,
            jt_prior_inv,
            err_prior,
            h_pp_schur,
            b_pp_schur,
            h_pp,
            b_pp,
            h_ll,
            b_ll,
            vertices,
            edges,
            vertex_to_edge,
            ordering_poses,
            ordering_landmarks,
            ordering_generic,
            idx_pose_vertices,
            idx_landmark_vertices,
            vertices_marg,
            b_debug,
            t_hessian_cost,
            out_file,
            n_iter,
        }
    }

    pub fn add_vertex(&mut self, vertex: Arc<Mutex<dyn Vertex>>) -> bool {
        let idx = vertex.lock().unwrap().id(); 
        if self.vertices.contains_key(&idx) {
            return false; 
        }
        self.vertices.insert(idx, vertex);
        true 
    }

    pub fn add_edge(&mut self, edge: Arc<Mutex<dyn Edge>>) -> bool {
        let idx = edge.clone().lock().unwrap().id(); 
        if self.edges.contains_key(&idx) {
            return false; 
        }
        self.edges.insert(idx, edge.clone()); 
        for vertex in edge.clone().lock().unwrap().get_vertices() {
            let idx = vertex.lock().unwrap().id();
            self.vertex_to_edge.entry(idx).or_insert_with(Vec::new).push(edge.clone()); 
        }
        true 
    }

    pub fn set_ordering(&mut self) {
        self.ordering_poses = 0; 
        self.ordering_landmarks = 0; 
        self.ordering_generic = 0; 

        for (key, value) in self.vertices.iter() {
            self.ordering_generic += value.lock().unwrap().local_dimension(); 
        }
    }

    pub fn compute_lambda_init_lm(&mut self) {
        println!("init lambda"); 

        self.ni = 2.0; 
        self.current_lambda = -1.0; 
        self.current_chi = 0.0; 
        for (key, value) in self.edges.iter() {
            let edge = value.lock().unwrap();
            self.current_chi += edge.chi2(); 
        }
        if self.err_prior.nrows() > 0 {
            self.current_chi += self.err_prior.norm(); 
        }

        self.stop_threshold_lm = 1e-6 * self.current_chi;  

        let mut max_diagonal: f64 = 0.0; 
        let size = self.hessian.ncols();
        assert_eq!(self.hessian.nrows(), size, "Hessian is not square"); 
        for i in 0..size {
            max_diagonal = max_diagonal.max(self.hessian[(0, 0)].abs()); 
        }
        let tau = 1e-5; 
        self.current_lambda = tau * max_diagonal; 
    }

    pub fn add_lambda_to_hessian_lm(&mut self) {
        let size = self.hessian.ncols(); 
        assert_eq!(self.hessian.nrows(), size, "Hessian is not square"); 
        for i in 0..size {
            self.hessian[(i, i)] += self.current_lambda; 
        }
    }

    pub fn remove_lambda_hessian_lm(&mut self) {
        let size = self.hessian.ncols(); 
        assert_eq!(self.hessian.nrows(), size, "Hessian is not square"); 
        for i in 0..size {
            self.hessian[(i, i)] -= self.current_lambda; 
        }
    }

    pub fn make_hessian(&mut self) {
        let mut hessian = MatXX::zeros(self.ordering_generic, self.ordering_generic); 
        let mut b = VecX::zeros(self.ordering_generic); 

        // iterate residual, compute jacobian, get H = J^T * J
        for (key, value) in self.edges.iter() {
            let mut edge = value.lock().unwrap(); 
            edge.compute_residual(); 
            edge.compute_jacobians(); 

            let jacobians = edge.get_jacobians(); 
            let vertices = edge.get_vertices(); 
            assert_eq!(jacobians.len(), vertices.len(), "Jacobians and vertices lengths do not match");
            for i in 0..vertices.len() {
                let v_i = vertices[i].lock().unwrap(); 
                if v_i.is_fixed() {
                    continue; 
                }
                let jacobian_i = jacobians[i]; 
                let index_i = v_i.ordering_id(); 
                let dim_i = v_i.local_dimension(); 

                let jtw = jacobian_i.transpose() * edge.get_information();
                for j in i..vertices.len() {
                    let jacobian_j; 
                    let index_j; 
                    let dim_j;
                    // avoid deadlock 
                    if i != j {
                        let v_j = vertices[j].lock().unwrap(); 
                        if v_j.is_fixed() {
                            continue; 
                        }
                        jacobian_j = jacobians[j]; 
                        index_j = v_j.ordering_id(); 
                        dim_j = v_j.local_dimension();
                    } else {
                        jacobian_j = jacobian_i.clone(); 
                        index_j = index_i; 
                        dim_j = dim_i;     
                    }
                    
                    let hessian_ij = jtw.clone() * jacobian_j; 
                    let mut subblock = hessian.view_mut((index_i, index_j), (dim_i, dim_j));
                    subblock += hessian_ij;   
                    if i != j {
                        // hessian is symmetric matrix 
                        let mut subblock = hessian.view_mut((index_j, index_i), (dim_j, dim_i));
                        subblock += hessian_ij.transpose();  
                    }               
                }
                let mut segment = b.view_mut((index_i, 0), (dim_i, 1));
                segment -= jtw.clone() * edge.get_residual(); 
            }
        }

        self.hessian = hessian; 
        self.b = b; 
        self.delta_x = VecX::zeros(self.ordering_generic);
    }

    pub fn update_states(&mut self) {
        for (key, value) in self.vertices.iter() {
            let mut vertex = value.lock().unwrap(); 
            let idx = vertex.ordering_id(); 
            let dim = vertex.local_dimension();
            let delta: VecX = self.delta_x.rows(idx, dim).into();
            vertex.plus(delta); 
        }
    }

    pub fn is_good_step_in_lm(&mut self) -> bool {
        let mut scale = 0.0; 
        scale = (self.delta_x.transpose() * (self.current_lambda * self.delta_x.clone() + self.b.clone())).into_scalar(); 
        scale += 1e-3; 

        let mut temp_chi = 0.0; 
        for (key, value) in self.edges.iter() {
            let mut edge = value.lock().unwrap(); 
            edge.compute_residual(); 
            temp_chi += edge.chi2(); 
        }

        let rho = (self.current_chi - temp_chi) / scale; 
        if rho > 0.0 && temp_chi.is_finite() {
            let mut alpha: f64 = 1.0 - (2.0 * rho - 1.0).powf(3.0); 
            alpha = alpha.min(2.0 / 3.0); 
            let scale_factor: f64 = alpha.max(1.0 / 3.0); 
            self.current_lambda *= scale_factor; 
            self.ni = 2.0;
            self.current_chi = temp_chi; 
            self.n_iter += 1; 
            // Write to the file
            writeln!(self.out_file, "{} {}", self.n_iter, self.current_lambda).expect("Should be able to write to file");
            // Also print to the console
            println!("n_iter {} current lambda {}", self.n_iter, self.current_lambda);  
            return true;           
        } else {
            self.current_lambda *= self.ni; 
            self.ni *= 2.0; 
            self.n_iter += 1; 
            // Write to the file
            writeln!(self.out_file, "{} {}", self.n_iter, self.current_lambda).expect("Should be able to write to file");
            // Also print to the console
            println!("n_iter {} current lambda {}", self.n_iter, self.current_lambda);  
            return false;      
        }
    }

    pub fn roll_back_states(&mut self) {
        for (key, value) in self.vertices.iter() {
            let mut vertex = value.lock().unwrap(); 
            let idx = vertex.ordering_id(); 
            let dim = vertex.local_dimension(); 
            let delta: VecX = self.delta_x.rows(idx, dim).into(); 
            vertex.plus(-1.0 * delta); 
        }
    }

    pub fn solve_linear_system(&mut self) {
        self.delta_x = self.hessian.clone().try_inverse().unwrap() * self.b.clone(); 
    }

    /// The method used to update lambda is Nielsen 
    pub fn solve(&mut self, iterations: usize) -> bool {
        if self.edges.is_empty() && self.vertices.is_empty() {
            return false; 
        }

        self.n_iter = 0; 

        self.set_ordering(); 
        self.make_hessian(); 
        self.compute_lambda_init_lm(); 
    
        let mut stop = false; 
        let mut iter: usize = 0; 

        self.n_iter += 1; 
        // Write to the file
        writeln!(self.out_file, "{} {}", self.n_iter, self.current_lambda).expect("Should be able to write to file");
        // Also print to the console
        println!("n_iter {} current lambda {}", self.n_iter, self.current_lambda);

        while !stop && iter < iterations {
            println!("iter {iter} chi {} lambda {}", self.current_chi, self.current_lambda); 

            let mut one_step_success = false; 
            let mut false_count = 0; 
            while !one_step_success {
                self.add_lambda_to_hessian_lm(); 
                self.solve_linear_system(); 
                self.remove_lambda_hessian_lm(); 

                // exit if delta x is small 
                if self.delta_x.norm() <= 1e-6 || false_count > 10 {
                    stop = true; 
                    break; 
                }

                self.update_states(); 
                one_step_success = self.is_good_step_in_lm(); 

                if one_step_success {
                    // make hessian at the updated linearization point 
                    self.make_hessian(); 
                    false_count = 0; 
                } else {
                    false_count += 1; 
                    self.roll_back_states(); 
                }
            }   
            iter += 1; 

            // if current chi decreases a lot then break 
            if self.current_chi.sqrt() <= self.stop_threshold_lm {
                stop = true; 
            }
        }

        true 
    }
}
