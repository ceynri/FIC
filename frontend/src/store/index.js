import Vue from 'vue';
import Vuex from 'vuex';

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    cssVars: {},
  },
  mutations: {
    setState(state, payload) {
      state[payload.state] = payload.value;
    },
  },
  actions: {},
  modules: {},
});
