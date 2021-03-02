import Vue from 'vue';
import VueRouter from 'vue-router';

import Home from '@/views/Home.vue';
import Demo from '@/views/Demo.vue';
import Compress from '@/views/Compress.vue';
import Decompress from '@/views/Decompress.vue';

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home,
  },
  {
    path: '/demo',
    name: 'Demo',
    component: Demo,
  },
  {
    path: '/compress',
    name: 'Compress',
    component: Compress,
  },
  {
    path: '/decompress',
    name: 'Decompress',
    component: Decompress,
  },
  {
    path: '/about',
    name: 'About',
    component: () => import('../views/About.vue'),
  },
];

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes,
});

export default router;
