use nalgebra as na; 
use std::sync::atomic::{Ordering, AtomicUsize};

use crate::nalgebra_types::*; 

pub trait Vertex {
    /// Constructor 
    fn new(dimension: usize, local_dimension: usize) -> Self; 
    /// Return the id of this vertex 
    fn id(&self) -> usize;
    /// Return the parameters 
    fn parameters(&self) -> VecX;
    /// Set the parameters 
    fn set_parameters(&mut self, params: VecX);
    /// Define the plus operation 
    fn plus(&mut self, delta: VecX); 
    /// Return the dimension 
    fn dimension(&self) -> usize;
    /// Return the local dimension 
    fn local_dimension(&self) -> usize;
    /// Return the id in jacobian
    fn ordering_id(&self) -> usize;
    /// Set the id in jacobian 
    fn set_ordering_id(&mut self, id: usize);
    /// Mark this vertex as fixed 
    fn set_fixed(&mut self, fixed: bool);
    /// Check whether this vertex is fixed 
    fn is_fixed(&self) -> bool;
    /// Get the type info 
    fn type_info(&self) -> String; 
}

// global variable ref 
// https://course.rs/advance/global-variable.html
static GLOBAL_VERTEX_ID: AtomicUsize = AtomicUsize::new(0);
const MAX_ID: usize = usize::MAX / 2;

fn generate_id() -> usize {
    let cur = GLOBAL_VERTEX_ID.load(Ordering::Relaxed); 
    if cur > MAX_ID {
        panic!("vertex ids overflow"); 
    } 
    GLOBAL_VERTEX_ID.fetch_add(1, Ordering::Relaxed); 
    let next = GLOBAL_VERTEX_ID.load(Ordering::Relaxed); 
    if next > MAX_ID {
        panic!("vertex ids overflowed"); 
    }
    next 
}

struct CurveFittingVertex {
    pub parameters: VecX, 
    pub local_dimension: usize, 
    pub id: usize, 
    pub ordering_id: usize, 
    pub fixed: bool, 
}

impl Vertex for CurveFittingVertex {
    fn new(dimension: usize, local_dimension: usize) -> Self {
        let parameters = na::DVector::from_element(dimension, 0.0); 
        let local_dimension = if local_dimension > 0 {local_dimension} else {dimension};
        let id = generate_id(); 
        let ordering_id = 0; 
        let fixed = false; 

        CurveFittingVertex {
            parameters, 
            local_dimension, 
            id, 
            ordering_id, 
            fixed, 
        }
    }

    /// Return the id of this vertex 
    fn id(&self) -> usize {
        self.id 
    }
    /// Return the parameters 
    fn parameters(&self) -> VecX {
        self.parameters.clone() 
    } 
    /// Set the parameters 
    fn set_parameters(&mut self, params: VecX) {
        self.parameters = params; 
    }
    /// Return the dimension 
    fn dimension(&self) -> usize {
        self.parameters.nrows()
    }
    /// Return the local dimension 
    fn local_dimension(&self) -> usize {
        self.local_dimension
    }
    /// Return the id in jacobian
    fn ordering_id(&self) -> usize {
        self.ordering_id
    }
    /// Set the id in jacobian 
    fn set_ordering_id(&mut self, id: usize) {
        self.ordering_id = id; 
    }
    /// Mark this vertex as fixed 
    fn set_fixed(&mut self, fixed: bool) {
        self.fixed = fixed; 
    }
    /// Check whether this vertex is fixed 
    fn is_fixed(&self) -> bool {
        self.fixed 
    }

    fn plus(&mut self, delta: VecX) {
        self.parameters += delta; 
    }

    fn type_info(&self) -> String {
        "curve_fitting_vertex".to_string()
    } 
}