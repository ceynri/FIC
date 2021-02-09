<template>
  <section class="decompressor page_frame">
    <header class="title_wrapper">
      <h1 class="title">Decompressor</h1>
      <div class="comment">Upload compressed images to decompress</div>
    </header>
    <Uploader accept=".fic" />
  </section>
</template>

<script>
import Uploader from '@/components/Uploader.vue';

export default {
  components: {
    Uploader,
  },
};
</script>

<style lang="scss" scoped>
.decompressor {
}
</style>
