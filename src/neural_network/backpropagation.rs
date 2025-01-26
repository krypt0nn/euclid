use super::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Information used for values backpropagation.
///
/// Stores AdamW optimizer info for `SIZE` parameters in `F` float type.
/// Uses cyclic learn rate scheduler with warmup stage.
///
/// If you plan to perform `N` backpropagation calls, then it's recommended to:
///
/// - Set `warmup_duration` from `0.05 * N` to `0.1 * N` (5-10% of total time).
/// - Set `cycle_duration` from `0.1 * N` to `0.2 * N` (10-20% of total time).
/// - Set `cycle_radius` from `3 * learn_rate` to `10 * learn_rate`.
///
/// This might seem weird to set large cycle radius, but its goal is to leave
/// the local minimum of the loss function for given parameters set, and large
/// learn rate changes are needed to achieve this.
///
/// It's not recommended to change AdamW optimizer values, defaults are found
/// to be most suitable for all the cases by many researchers.
pub struct Backpropagation<const SIZE: usize, F: Float> {
    /// Current backpropagation timestep.
    ///
    /// Shows amount of already passed backpropagations.
    timestep: u32,

    /// AdamW backpropagation optimization momentums.
    adamw_m: [F; SIZE],

    /// AdamW backpropagation optimization squared momentums.
    adamw_v: [F; SIZE],

    /// AdamW backpropagation optimization hyperparameter.
    ///
    /// Beta 1 controls the exponential moving average of the gradient.
    /// A high value of beta 1 means that the optimizer will give more weight
    /// to the previous gradients, while a low value means that the optimizer
    /// will give more weight to the current gradient.
    ///
    /// Recommended values: 0.9, 0.95, 0.99.
    adamw_beta1: F,

    /// AdamW backpropagation optimization hyperparameter.
    ///
    /// Beta 2 controls the exponential moving average of the squared gradient.
    /// A high value of beta 2 means that the optimizer will give more weight
    /// to the previous squared gradients, while a low value means that the optimizer
    /// will give more weight to the current squared gradient.
    ///
    /// Recommended values: 0.999, 0.995, 0.99.
    adamw_beta2: F,

    /// AdamW backpropagation optimization hyperparameter.
    ///
    /// Weight decay is a regularization technique that adds a penalty term
    /// to the loss function to discourage large weights. A high value
    /// of weight decay means that the optimizer will penalize large weights
    /// more heavily.
    ///
    /// Recommended values: 0.01, 0.001, 0.0001.
    adamw_lambda: F,

    /// Amount of backpropagation timesteps to reach the target learn rate value.
    ///
    /// If set to 0 then no warmup will be applied.
    warmup_duration: u32,

    /// Maximal difference from the target learn rate value.
    ///
    /// Actual learn rate will vary in range `[target - radius, target + radius]`.
    ///
    /// If set to 0 then no cyclic schedule will be applied.
    cycle_radius: F,

    /// Amount of timesteps of backpropagation for going full cicle
    /// of learning rate changing.
    ///
    /// If set to 0 then no cyclic schedule will be applied.
    cycle_period: u32,

    /// Target learn rate of the backpropagation.
    ///
    /// It is different from the actual learn rate because we use cyclic
    /// learn rate update schedule and warmup phase. During warmup learning
    /// rate will slowly increase from `F::EPSILON` to the current value,
    /// and then it will start going lower and higher according to the
    /// cyclic schedule around the target value.
    ///
    /// ```text,ignore
    ///                learn rate
    ///                    ^
    ///                    |        . .         . .
    /// target learn rate  +      .     .     .     .     .
    ///                    |    +         . .         . .
    ///                    |   /
    ///                    |  /
    ///                    | /
    ///                    +-------------------------------> time
    ///                     ^^^^^^ warmup
    /// ```
    learn_rate: F
}

impl<const SIZE: usize, F: Float> Default for Backpropagation<SIZE, F> {
    fn default() -> Self {
        Self {
            timestep: 0,

            adamw_m: [F::ZERO; SIZE],
            adamw_v: [F::ZERO; SIZE],

            // Recommended defaults for AdamW optimizer.
            adamw_beta1: F::from_float(0.9),
            adamw_beta2: F::from_float(0.999),
            adamw_lambda: F::from_float(0.01),

            // Do not do learn rate warmup.
            warmup_duration: 0,

            // Do not do cyclic schedule for the learn rate.
            cycle_radius: F::ZERO,
            cycle_period: 0,

            // Some default learn rate value generated by LLM with this reasoning:
            //
            // - Works well as a max learning rate when using AdamW.
            // - Acts as a balanced starting point for cyclic schedules.
            // - Empirically validated in many setups.
            learn_rate: F::from_float(0.0003)
        }
    }
}

impl<const SIZE: usize, F: Float> Backpropagation<SIZE, F> {
    /// Try to find best backpropagation parameters by slowly increasing
    /// the learning rate and testing results with provided loss function.
    ///
    /// This method will try to predict the most suitable learn rate value
    /// for default AdamW optimizer parameters. It will not use warmup stage
    /// and cyclic update schedule. You should setup them yourself.
    ///
    /// `exploration_loss` should apply given backpropagation policy
    /// to some model and return its loss value.
    ///
    /// Make sure to use the same model with its *initial* state
    /// to compare impact of learn rate change.
    pub fn explore(exploration_loss: impl Fn(&Self) -> f64) -> Self {
        const SMOOTH_WINDOW: usize = 10;
        const SMOOTH_WINDOW_FLOAT: f64 = SMOOTH_WINDOW as f64;

        let mut learn_rate = F::EPSILON;
        let learn_rate_step = F::from_float(2.0);

        // Create backpropagation policy with default values and set
        // its learn rate to zero.
        let mut backpropagation = Self::default()
            .with_learn_rate(learn_rate);

        // Remember last 10 loss values.
        let mut losses = [1.0; SMOOTH_WINDOW];
        let mut avg_loss = 0.0;

        #[allow(clippy::needless_range_loop)]
        // Fill first 10 loss values.
        for i in 0..10 {
            // Calculate loss for the current learn rate value.
            losses[i] = exploration_loss(&backpropagation) / SMOOTH_WINDOW_FLOAT;

            avg_loss += losses[i];

            // Increase learn rate and update the policy.
            learn_rate *= learn_rate_step;
            backpropagation = backpropagation.with_learn_rate(learn_rate);
        }

        // Keep increasing learn rate until we see the spike in loss value.
        while learn_rate != F::MAX {
            // Calculate loss for the current learn rate value.
            let mut loss = exploration_loss(&backpropagation);

            // Break exploration if found loss is greater than 1.5 of moving average.
            if loss > avg_loss * 1.5 {
                // Use quarter of found learn rate for the final result.
                backpropagation = backpropagation.with_learn_rate(learn_rate * F::HALF * F::HALF);

                break;
            }

            // Update moving average.
            loss /= SMOOTH_WINDOW_FLOAT;

            avg_loss -= losses[0];
            avg_loss += loss;

            for j in 0..9 {
                losses[j] = losses[j + 1];
            }

            losses[9] = loss;

            // Increase learn rate and update the policy.
            learn_rate *= learn_rate_step;
            backpropagation = backpropagation.with_learn_rate(learn_rate);
        }

        backpropagation
    }

    #[inline]
    /// Change AdamW beta1 parameter.
    pub fn with_adamw_beta1(mut self, adamw_beta1: F) -> Self {
        self.adamw_beta1 = adamw_beta1;

        self
    }

    #[inline]
    /// Change AdamW beta2 parameter.
    pub fn with_adamw_beta2(mut self, adamw_beta2: F) -> Self {
        self.adamw_beta2 = adamw_beta2;

        self
    }

    #[inline]
    /// Change AdamW lambda parameter.
    pub fn with_adamw_lambda(mut self, adamw_lambda: F) -> Self {
        self.adamw_lambda = adamw_lambda;

        self
    }

    #[inline]
    /// Change target learn rate of the backpropagation.
    pub fn with_learn_rate(mut self, learn_rate: F) -> Self {
        self.learn_rate = learn_rate;

        self
    }

    #[inline]
    /// Change warmup stage duration in backpropagation timesteps.
    pub fn with_warmup_duration(mut self, warmup_duration: u32) -> Self {
        self.warmup_duration = warmup_duration;

        self
    }

    #[inline]
    /// Change cycle radius of the cyclic learn rate schedule.
    pub fn with_cycle_radius(mut self, cycle_radius: F) -> Self {
        self.cycle_radius = cycle_radius;

        self
    }

    #[inline]
    /// Change cycle period of the cyclic learn rate schedule in backpropagation timesteps.
    pub fn with_cycle_period(mut self, cycle_period: u32) -> Self {
        self.cycle_period = cycle_period;

        self
    }

    /// Backpropagate given values using calculated gradients.
    pub fn backpropagate(&mut self, mut values: [F; SIZE], gradients: [F; SIZE]) -> [F; SIZE] {
        self.timestep += 1;

        // Learn rate warmup.
        let learn_rate = if self.timestep < self.warmup_duration {
            self.learn_rate * F::from_float(self.timestep as f32 / self.warmup_duration as f32)
        }

        // Either period or radius of the cycle are zero, which mean
        // we can't apply cyclic schedule to the learn rate.
        else if self.cycle_period == 0 || self.cycle_radius == F::ZERO {
            self.learn_rate
        }

        // Cyclic schedule.
        else {
            // Amount of passed timesteps within the cycle.
            let cycle_steps = (self.timestep - self.warmup_duration) % self.cycle_period;

            // Percent of cycle finishing.
            let cycle_value = cycle_steps as f32 / self.cycle_period as f32;

            // Calculate current sine value, scale it by the radius and add it to the learn rate.
            self.learn_rate + self.cycle_radius * F::from_float((2.0 * std::f32::consts::PI * cycle_value).sin())
        };

        // Calculate AdamW beta powers.
        let adamw_inv_beta1_t = F::ONE - self.adamw_beta1.powi(self.timestep as i32);
        let adamw_inv_beta2_t = F::ONE - self.adamw_beta2.powi(self.timestep as i32);

        // Update given values.
        for i in 0..SIZE {
            // Update AdamW moving averages.
            self.adamw_m[i] = self.adamw_beta1 * self.adamw_m[i] + (F::ONE - self.adamw_beta1) * gradients[i];
            self.adamw_v[i] = self.adamw_beta2 * self.adamw_v[i] + (F::ONE - self.adamw_beta2) * gradients[i].powi(2);

            // Calculate their weighted values.
            let adamw_weighted_m = self.adamw_m[i] / adamw_inv_beta1_t;
            let adamw_weighted_v = self.adamw_v[i] / adamw_inv_beta2_t;

            // Update value using gradient and calculated AdamW values.
            values[i] -= learn_rate * adamw_weighted_m / (adamw_weighted_v.sqrt() + F::EPSILON) + learn_rate * self.adamw_lambda * values[i];
        }

        values
    }
}
