import Vue from 'vue';

import router from './router';
import './mixin';
import './plugins';

import App from './App.vue';

import '@/styles/base.scss';

Vue.config.productionTip = false;

new Vue({
  router,
  render: (h) => h(App),
}).$mount('#app');
