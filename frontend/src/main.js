import Vue from 'vue';

import OverlayScrollbars from 'overlayscrollbars';
import { OverlayScrollbarsPlugin, OverlayScrollbarsComponent } from 'overlayscrollbars-vue';

import 'overlayscrollbars/css/OverlayScrollbars.css';
import '@/styles/base.scss';

import IconBase from '@/components/IconBase.vue';

import App from './App.vue';
import router from './router';
import store from './store';
import mixin from './mixin';

Vue.use(OverlayScrollbarsPlugin);
Vue.component('overlay-scrollbars', OverlayScrollbarsComponent);
Vue.component('IconBase', IconBase);

Vue.mixin(mixin);

Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount('#app');

OverlayScrollbars(document.body, {
  nativeScrollbarsOverlaid: {
    initialize: false,
  },
});
