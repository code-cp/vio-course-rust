use std::sync::{Arc, Mutex}; 
use nalgebra as na; 
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::vertex::*; 
use crate::nalgebra_types::*; 

pub trait Edge {
    fn id(&self) -> usize; 
    fn add_vertex(&mut self, vertex: Arc<Mutex<dyn Vertex>>); 
    fn set_vertex(&mut self, vertices: Vec<Arc<Mutex<dyn Vertex>>>); 
    fn get_vertex(&self, i: usize) -> Arc<Mutex<dyn Vertex>>; 
    fn get_vertices(&self) -> Vec<Arc<Mutex<dyn Vertex>>>; 
    fn num_vertices(&self) -> usize; 
    fn type_info(&self) -> String; 
    fn compute_residual(&mut self); 
    fn compute_jacobians(&mut self); 
    fn set_information(&mut self, information: MatXX); 
    fn get_information(&self) -> MatXX; 
    fn set_observation(&mut self, observation: VecX); 
    fn get_observation(&self) -> VecX; 
    fn check_valid(&self) -> bool; 
}

// global variable ref 
// https://course.rs/advance/global-variable.html
static GLOBAL_EDGE_ID: AtomicUsize = AtomicUsize::new(0);
const MAX_ID: usize = usize::MAX / 2;

fn generate_id() -> usize {
    let cur = GLOBAL_EDGE_ID.load(Ordering::Relaxed); 
    if cur > MAX_ID {
        panic!("vertex ids overflow"); 
    } 
    GLOBAL_EDGE_ID.fetch_add(1, Ordering::Relaxed); 
    let next = GLOBAL_EDGE_ID.load(Ordering::Relaxed); 
    if next > MAX_ID {
        panic!("vertex ids overflowed"); 
    }
    next 
}

pub struct CurveFittingEdge {
    pub id: usize, 
    pub ordering_id: usize, 
    pub verticies_types: Vec<String>, 
    pub vertices: Vec<Arc<Mutex<dyn Vertex>>>, 
    pub residual: VecX, 
    pub jacobians: Vec<RowVec3>,
    pub information: MatXX, 
    pub observation: VecX, 
    pub x: f64, 
    pub y: f64, 
}

impl CurveFittingEdge {
    fn new(residual_dimension: usize, num_verticies: usize, verticies_types: Vec<String>) -> Self {
        let residual = VecX::from_element(residual_dimension, 0.0); 
        let jacobians: Vec<RowVec3> = Vec::with_capacity(num_verticies); 
        let id = generate_id(); 
        let ordering_id = 0; 
        let information = MatXX::identity(residual_dimension, residual_dimension); 
        let observation = VecX::from_element(residual_dimension, 0.0); 
        let vertices: Vec<Arc<Mutex<dyn Vertex>>> = Vec::with_capacity(num_verticies); 
        let x = 0.0; 
        let y = 0.0; 

        CurveFittingEdge {
            residual, 
            verticies_types, 
            vertices, 
            jacobians, 
            id, 
            ordering_id, 
            information, 
            observation, 
            x, 
            y, 
        }
    }
}

impl Edge for CurveFittingEdge {
    fn id(&self) -> usize {
        self.id 
    }

    fn add_vertex(&mut self, vertex: Arc<Mutex<dyn Vertex>>) {
        self.vertices.push(vertex); 
    }

    fn set_vertex(&mut self, vertices: Vec<Arc<Mutex<dyn Vertex>>>) {
        self.vertices = vertices; 
    }

    fn get_vertex(&self, i: usize) -> Arc<Mutex<dyn Vertex>> {
        self.vertices[i].clone()
    }

    fn get_vertices(&self) -> Vec<Arc<Mutex<dyn Vertex>>> {
        self.vertices.clone()
    }

    fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    fn type_info(&self) -> String {
        "CurveFittingEdge".to_string()
    }

    fn compute_residual(&mut self) {
        let abc = self.vertices[0].lock().unwrap().parameters(); 
        let r = f64::exp(abc[0] * self.x.powf(2.0) + abc[1] * self.x + abc[2]) - self.y; 
        self.residual[0] = r; 
    }

    fn compute_jacobians(&mut self) {
        let abc = self.vertices[0].lock().unwrap().parameters(); 
        let exp_y = f64::exp(abc[0] * self.x.powf(2.0) + abc[1] * self.x + abc[2]); 
        let jaco_abc = na::RowVector3::new(
            exp_y * self.x.powf(2.0), 
            exp_y * self.x, 
            exp_y, 
        );
        self.jacobians.push(jaco_abc); 
    }

    fn set_information(&mut self, information: MatXX) {
        self.information = information; 
    }

    fn get_information(&self) -> MatXX {
        self.information.clone() 
    }

    fn set_observation(&mut self, observation: VecX) {
        self.observation = observation; 
    }

    fn get_observation(&self) -> VecX {
        self.observation.clone()
    }

    fn check_valid(&self) -> bool {
        if self.verticies_types.len() == 0 {
            return true; 
        }

        for i in 0..self.verticies_types.len() {
            let vertex = self.vertices[i].lock().unwrap(); 
            if self.verticies_types[i] != vertex.type_info() {
                println!("vertex type should be {}, but is set to {}", 
                    self.verticies_types[i], 
                    vertex.type_info(), 
                );
                return false; 
            }
        }

        return true; 
    }
}