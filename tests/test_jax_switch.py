import jax
from jax import lax
from flax import struct
from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX

@struct.dataclass
class Holding:
    nothing = 0
    onion = 1
    plate = 2
    dish = 3 # Completed soup


@jax.jit
def get_holding(inv_idx):
    '''Maps inventory index to Holding struct value'''
    holding = lax.cond(
        inv_idx == OBJECT_TO_INDEX['empty'],
        lambda _: Holding.nothing,
        lambda _: lax.cond(
            inv_idx == OBJECT_TO_INDEX['onion'],
            lambda _: Holding.onion,
            lambda _: lax.cond(
                inv_idx == OBJECT_TO_INDEX['plate'],
                lambda _: Holding.plate,
                lambda _: lax.cond(
                    inv_idx == OBJECT_TO_INDEX['dish'],
                    lambda _: Holding.dish,
                    lambda _: -1, # non-supported index
                    None),
                None),
            None),
        None)
    return holding

if __name__ == "__main__":
    inv_idx = 5 # choices: 1, 3, 5, 9
    print(get_holding(inv_idx))

